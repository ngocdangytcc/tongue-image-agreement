"""P3 FINAL — full 2,452-image run, both epoch budgets, with bootstrap CIs.

    %run /content/drive/MyDrive/tongue-ai-work/run_final.py

Standalone and resume-safe. Fixes the defect in notebook cell 3, which derived the
PNG name by replacing only a HEIC/heic/heif suffix; any other extension produced a
path that did not exist and the image was dropped silently, costing 33 images and
7 patients. Deriving the name from the stem covers every extension, and the
reconciliation below now reports drops instead of hiding them.

Trains 6 and 12 epochs so the robustness comparison sits on the same dataset as the
headline result. Each (epochs, fold) is checkpointed to Drive before the next starts;
re-running after a disconnect resumes rather than restarts.
"""

import hashlib
import json
import random
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

WORK = Path("/content/drive/MyDrive/tongue-ai-work")
OUT = WORK / "p3_final"
FOLD_DIR = OUT / "folds"
OUT.mkdir(parents=True, exist_ok=True)
FOLD_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE, IMG_SIZE, BATCH, FOLDS, LR = "efficientnet_b0", 384, 32, 5, 3e-4
SEED, N_BOOT, NUM_WORKERS = 20260915, 1000, 4
EPOCH_GRID = [6, 12]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORICAL = OrderedDict([
    ("Tongue_Color",      ["Pale", "Pale_red", "Red", "Bluish_purple"]),
    ("Tongue_Shape",      ["Thin", "Normal", "Enlarged"]),
    ("Tongue_Moisture",   ["Dry", "Moist", "Wet"]),
    ("Coating_Color",     ["White", "White_yellow", "Yellow", "Grayish_black"]),
    ("Coating_Thickness", ["Thin", "Thick"]),
    ("Tongue_Movement",   ["Flexible", "Trembling", "Flaccid", "Stiff"]),
])
BINARY = ["Tongue_Swollen", "Coating_Peeled", "Coating_Rotten", "Coating_Greasy",
          "Tongue_Scalloped", "Tongue_Cracked", "Tongue_Prickly",
          "Tongue_Stasis_Spots", "Tongue_Sublingual_Veins"]
NEG_CTRL = {"Tongue_Movement", "Tongue_Sublingual_Veins"}
TASKS = [(c, len(v)) for c, v in CATEGORICAL.items()] + [(c, 1) for c in BINARY]


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- 1. local cache
PNG = Path("/content/png_512")
src = WORK / "png_512" / "dorsal"
dst = PNG / "dorsal"
dst.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("*.png"))
todo = [f for f in files if not (dst / f.name).exists()]
print(f"PNG cache: {len(files)} in Drive, {len(todo)} to copy locally")
if todo:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda f: shutil.copy2(f, dst / f.name), todo))
    print(f"  copied in {time.time() - t0:.0f}s")
assert len(list(dst.glob("*.png"))) == len(files), "local cache incomplete"

# ---------------------------------------------------- 2. data, with the fix applied
man = pd.read_csv(WORK / "manifest.csv")
cohort = pd.read_csv(WORK / "Data.csv", encoding="utf-8-sig")

d = man[(man.kind == "image") & (man.parse_ok == 1) & (man.view == "dorsal")].copy()
d["patient_id"] = d.patient_id.astype(int)
# THE FIX: derive from the stem, so .JPG/.jpeg/.png resolve as well as .HEIC
d["png_path"] = d.filename.map(lambda f: str(dst / (Path(f).stem + ".png")))
before_n, before_p = len(d), d.patient_id.nunique()
exists = d.png_path.map(lambda p: Path(p).exists())
dropped = d[~exists]
d = d[exists].copy()

print(f"\nreconciliation: {before_n} dorsal rows in manifest -> {len(d)} with a PNG")
print(f"  patients: {before_p} -> {d.patient_id.nunique()}")
if len(dropped):
    print(f"  !! {len(dropped)} DROPPED (no PNG found):")
    print(dropped[["filename", "png_path"]].head(20).to_string(index=False))
else:
    print("  no images dropped")

d["month"] = pd.to_datetime(d.exif_datetime, format="%Y:%m:%d %H:%M:%S",
                            errors="coerce").dt.to_period("M").astype(str)
d.loc[d.month == "NaT", "month"] = "unknown"
keep = ["Patient_ID", "Atrophic_Gastritis"] + list(CATEGORICAL) + BINARY
d = d.merge(cohort[keep], left_on="patient_id", right_on="Patient_ID")
for col, lv in CATEGORICAL.items():
    d[col + "__y"] = d[col].map({v: i for i, v in enumerate(lv)}).astype("Int64")
for col in BINARY:
    d[col + "__y"] = d[col].astype("Int64")
print(f"\nfinal dataset: {len(d)} images, {d.patient_id.nunique()} patients")

# ---------------------------------------------------------------- 3. folds
d["strat"] = d.arm_folder.astype(str) + "|" + d.month.astype(str)
vc = d.strat.value_counts()
d.loc[d.strat.isin(vc[vc < FOLDS * 2].index), "strat"] = "rare"
sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
d["fold"] = -1
for k, (_, va) in enumerate(sgkf.split(d, d.strat, groups=d.patient_id)):
    d.iloc[va, d.columns.get_loc("fold")] = k
for k in range(FOLDS):
    assert not (set(d[d.fold != k].patient_id) & set(d[d.fold == k].patient_id)), f"LEAK {k}"
print("patient-level leakage check: PASS")
print(d.groupby("fold").agg(images=("png_path", "size"), patients=("patient_id", "nunique")))

# ---------------------------------------------------------------- 4. model
TRAIN_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
    T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.06, hue=0.015),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
EVAL_TF = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class TongueDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.open(r.png_path).convert("RGB"))
        ys, ms = [], []
        for c, _ in TASKS:
            v = r[c + "__y"]
            ms.append(0.0 if pd.isna(v) else 1.0)
            ys.append(0 if pd.isna(v) else int(v))
        return x, torch.tensor(ys), torch.tensor(ms), int(r.patient_id)


class MultiHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.heads = nn.ModuleList([nn.Linear(self.enc.num_features, n) for _, n in TASKS])
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        z = self.drop(self.enc(x))
        return [h(z) for h in self.heads]


def multitask_loss(outs, y, m):
    total = 0.0
    for j, (c, n) in enumerate(TASKS):
        mj = m[:, j]
        if mj.sum() == 0:
            continue
        l = (F.binary_cross_entropy_with_logits(outs[j].squeeze(1), y[:, j].float(),
                                                reduction="none") if n == 1
             else F.cross_entropy(outs[j], y[:, j], reduction="none"))
        total = total + (l * mj).sum() / mj.sum()
    return total / len(TASKS)


@torch.no_grad()
def predict(model, loader):
    model.eval(); P = [[] for _ in TASKS]; pids = []
    for x, _, _, pid in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            outs = model(x)
        for j, (c, n) in enumerate(TASKS):
            o = outs[j].float()
            P[j].append((torch.sigmoid(o).squeeze(1) if n == 1
                         else torch.softmax(o, 1)).cpu().numpy())
        pids.append(np.asarray(pid))
    return [np.concatenate(p) for p in P], np.concatenate(pids)


# ---------------------------------------------------------------- 5. train
hashes = {}
for EPOCHS in EPOCH_GRID:
    CFG = {"backbone": BACKBONE, "img": IMG_SIZE, "epochs": EPOCHS, "folds": FOLDS,
           "lr": LR, "seed": SEED, "batch": BATCH, "n_images": int(len(d))}
    h = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:12]
    hashes[EPOCHS] = h
    print(f"\n{'='*60}\n{EPOCHS} epochs  |  config {h}\n{'='*60}", flush=True)
    t0 = time.time()
    for k in range(FOLDS):
        fp = FOLD_DIR / f"fold{k}_{h}.npz"
        if fp.exists():
            print(f"fold {k}: cached, skipping", flush=True); continue
        seed_all(SEED + k)
        tr, va = d[d.fold != k], d[d.fold == k]
        dl_tr = DataLoader(TongueDS(tr, TRAIN_TF), batch_size=BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        dl_va = DataLoader(TongueDS(va, EVAL_TF), batch_size=BATCH * 2, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
        model = MultiHead(BACKBONE).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=EPOCHS * len(dl_tr), pct_start=0.25)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
        for ep in range(EPOCHS):
            model.train(); run = 0.0
            for x, y, m, _ in dl_tr:
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    loss = multitask_loss(model(x), y, m)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                run += loss.item()
            print(f"  fold {k} ep {ep+1}/{EPOCHS} loss {run/len(dl_tr):.4f}", flush=True)
        P, pids = predict(model, dl_va)
        np.savez_compressed(fp, pid=pids, **{f"t{j}": P[j] for j in range(len(TASKS))})
        torch.save(model.state_dict(), FOLD_DIR / f"weights{k}_{h}.pt")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"fold {k} saved ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- 6. metrics + CI
truth = d.drop_duplicates("patient_id").set_index("patient_id")


def evaluate(h):
    pids, per = [], [[] for _ in TASKS]
    for k in range(FOLDS):
        z = np.load(FOLD_DIR / f"fold{k}_{h}.npz")
        pids.append(z["pid"])
        for j in range(len(TASKS)):
            per[j].append(z[f"t{j}"])
    pid = np.concatenate(pids)
    res = {}
    for j, (name, n_cls) in enumerate(TASKS):
        p = np.concatenate(per[j])
        df = pd.DataFrame({"pid": pid})
        if p.ndim == 1:
            df["p0"] = p
        else:
            for i in range(p.shape[1]):
                df[f"p{i}"] = p[:, i]
        agg = df.groupby("pid").mean()
        y = truth.loc[agg.index, name + "__y"]
        ok = y.notna().to_numpy()
        yv = y[ok].astype(int).to_numpy()
        P = agg.to_numpy()[ok]
        if n_cls == 1:
            pr = P[:, 0]
            thr = float(np.quantile(pr, 1 - yv.mean()))
            yh = (pr >= thr).astype(int)
            try:
                auc = float(roc_auc_score(yv, pr))
            except Exception:
                auc = float("nan")
            extra = dict(prevalence=round(float(yv.mean()), 4), n_positive=int(yv.sum()),
                         sensitivity=round(float(recall_score(yv, yh, zero_division=0)), 4),
                         specificity=round(float(recall_score(1-yv, 1-yh, zero_division=0)), 4),
                         threshold=round(thr, 4))
        else:
            yh = P.argmax(1)
            try:
                auc = float(roc_auc_score(yv, P, multi_class="ovr", average="macro")
                            if n_cls > 2 else roc_auc_score(yv, P[:, 1]))
            except Exception:
                auc = float("nan")
            extra = dict(n_classes=n_cls,
                         majority_baseline=round(float(pd.Series(yv).value_counts(normalize=True).max()), 4),
                         accuracy=round(float(accuracy_score(yv, yh)), 4),
                         macro_f1=round(float(f1_score(yv, yh, average="macro", zero_division=0)), 4))
        rng = np.random.default_rng(SEED)
        ks, au = [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(yv), len(yv))
            yb = yv[idx]
            if len(np.unique(yb)) < 2:
                continue
            ks.append(cohen_kappa_score(yb, yh[idx]))
            try:
                au.append(roc_auc_score(yb, P[idx, 0]) if n_cls == 1 else
                          (roc_auc_score(yb, P[idx, 1]) if n_cls == 2 else
                           roc_auc_score(yb, P[idx], multi_class="ovr", average="macro")))
            except Exception:
                pass
        ci = lambda v: (round(float(np.percentile(v, 2.5)), 3),
                        round(float(np.percentile(v, 97.5)), 3)) if v else (None, None)
        res[name] = dict(n=int(ok.sum()),
                         kappa=round(float(cohen_kappa_score(yv, yh)), 4), kappa_ci=ci(ks),
                         auc=None if np.isnan(auc) else round(auc, 4), auc_ci=ci(au),
                         negative_control=name in NEG_CTRL, **extra)
    return res


report = {"study": "P3-final", "generated_utc": pd.Timestamp.utcnow().isoformat(),
          "device": DEVICE, "n_boot": N_BOOT, "seed": SEED,
          "data": {"images": int(len(d)), "patients": int(d.patient_id.nunique()),
                   "images_dropped": int(len(dropped)), "view": "dorsal"},
          "config_hashes": hashes, "configs": {}}

for EPOCHS, h in hashes.items():
    res = evaluate(h)
    report["configs"][f"{EPOCHS}ep"] = {"config_hash": h, "per_sign": res}
    readable = [v["kappa"] for k, v in res.items() if not v["negative_control"]]
    report["configs"][f"{EPOCHS}ep"]["median_kappa_readable"] = round(float(np.median(readable)), 4)
    print(f"\n=== {EPOCHS} epochs — kappa [95% CI] ===")
    rows = [{"sign": k, "ctrl": "NEG" if v["negative_control"] else "",
             "n_pos": v.get("n_positive", "-"), "kappa": v["kappa"],
             "95% CI": f"{v['kappa_ci'][0]}-{v['kappa_ci'][1]}", "auc": v["auc"]}
            for k, v in res.items()]
    print(pd.DataFrame(rows).sort_values("kappa", ascending=False).to_string(index=False))
    print(f"median kappa, 13 readable: {np.median(readable):.3f}")

(OUT / "report_p3_final.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8")
print("\n" + "=" * 66)
print("  PASTE EVERYTHING BELOW THIS LINE BACK INTO THE CHAT")
print("=" * 66)
print(json.dumps(report, indent=2, default=str))
print("=" * 66)
