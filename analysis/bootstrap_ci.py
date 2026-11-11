"""P3 — patient-level bootstrap confidence intervals for every tongue sign.

Run inside Colab with Drive mounted:

    exec(open('/content/drive/MyDrive/tongue-ai-work/bootstrap_ci.py').read())

Standalone by design: it reads the saved fold .npz files and Data.csv directly and
depends on no notebook variable, so it survives a runtime recycle and needs no GPU.
Every configuration it finds in p3_phenotype/folds/ is analysed and compared.

Thresholds for binary signs are fixed at the full-sample prevalence-matched value and
held constant across resamples: the bootstrap is meant to quantify uncertainty in the
performance of one fixed decision rule, not of a rule that re-tunes itself on every
resample, which would understate the interval.
"""

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

WORK = Path("/content/drive/MyDrive/tongue-ai-work")
OUT = WORK / "p3_phenotype"
FOLD_DIR = OUT / "folds"
N_BOOT = 1000
SEED = 20260915

# Must match the notebook's TASKS order exactly: the .npz keys are t0..t14 in this order.
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

cohort = pd.read_csv(WORK / "Data.csv", encoding="utf-8-sig")
truth = cohort.set_index("Patient_ID")


def load_config(cfg_hash):
    """Aggregate a configuration's out-of-fold predictions to one row per patient."""
    files = sorted(FOLD_DIR.glob(f"fold*_{cfg_hash}.npz"))
    pids, per_task = [], [[] for _ in TASKS]
    for f in files:
        z = np.load(f)
        pids.append(z["pid"])
        for j in range(len(TASKS)):
            per_task[j].append(z[f"t{j}"])
    pid = np.concatenate(pids)
    out = {}
    for j, (name, n) in enumerate(TASKS):
        p = np.concatenate(per_task[j])
        df = pd.DataFrame({"pid": pid})
        if p.ndim == 1:
            df["p0"] = p
        else:
            for i in range(p.shape[1]):
                df[f"p{i}"] = p[:, i]
        # a patient contributes two dorsal photographs; the clinical unit is the patient
        out[name] = df.groupby("pid").mean()
    return out, len(files)


def evaluate(name, n_cls, agg):
    """Point estimate plus patient-level bootstrap CI for one sign."""
    if name in CATEGORICAL:
        y = truth.loc[agg.index, name].map(
            {v: i for i, v in enumerate(CATEGORICAL[name])})
    else:
        y = truth.loc[agg.index, name]
    ok = y.notna().to_numpy()
    yv = y[ok].astype(int).to_numpy()
    P = agg.to_numpy()[ok]

    if n_cls == 1:
        p = P[:, 0]
        thr = float(np.quantile(p, 1 - yv.mean()))   # fixed across resamples
        yh = (p >= thr).astype(int)
        point_k = cohen_kappa_score(yv, yh)
        try:
            point_a = roc_auc_score(yv, p)
        except Exception:
            point_a = np.nan
    else:
        yh = P.argmax(1)
        thr = None
        point_k = cohen_kappa_score(yv, yh)
        try:
            point_a = (roc_auc_score(yv, P, multi_class="ovr", average="macro")
                       if n_cls > 2 else roc_auc_score(yv, P[:, 1]))
        except Exception:
            point_a = np.nan

    rng = np.random.default_rng(SEED)
    n = len(yv)
    ks, aucs = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = yv[idx]
        if len(np.unique(yb)) < 2:
            continue                       # a resample with one class carries no information
        ks.append(cohen_kappa_score(yb, yh[idx]))
        try:
            if n_cls == 1:
                aucs.append(roc_auc_score(yb, P[idx, 0]))
            elif n_cls == 2:
                aucs.append(roc_auc_score(yb, P[idx, 1]))
            else:
                aucs.append(roc_auc_score(yb, P[idx], multi_class="ovr", average="macro"))
        except Exception:
            pass

    def ci(v):
        return (round(float(np.percentile(v, 2.5)), 3),
                round(float(np.percentile(v, 97.5)), 3)) if v else (None, None)

    return {
        "n": int(ok.sum()),
        "prevalence": round(float(yv.mean()), 4) if n_cls == 1 else None,
        "n_positive": int(yv.sum()) if n_cls == 1 else None,
        "kappa": round(float(point_k), 4),
        "kappa_ci": ci(ks),
        "kappa_ci_width": round(ci(ks)[1] - ci(ks)[0], 3) if ks else None,
        "auc": None if np.isnan(point_a) else round(float(point_a), 4),
        "auc_ci": ci(aucs),
        "threshold": None if thr is None else round(thr, 4),
        "negative_control": name in NEG_CTRL,
        "boot_valid": len(ks),
    }


hashes = sorted({m.group(1) for m in
                 (re.search(r"fold\d+_([0-9a-f]{12})\.npz", f.name)
                  for f in FOLD_DIR.glob("fold*.npz")) if m})
if not hashes:
    raise SystemExit(f"no fold files found in {FOLD_DIR}")

label = {}
for h in hashes:
    rep = OUT / f"report_p3_{h}.json"
    ep = None
    for cand in [rep, OUT / "report_p3.json", OUT / "report_p3_e12.json"]:
        if cand.exists():
            try:
                r = json.load(open(cand, encoding="utf-8"))
                if r.get("config", {}).get("config_hash") == h:
                    ep = r["config"]["epochs"]
                    break
            except Exception:
                pass
    label[h] = f"{ep}ep" if ep else h[:6]

results = {}
for h in hashes:
    agg, n_folds = load_config(h)
    print(f"\n=== config {h}  ({label[h]}, {n_folds} folds, "
          f"{len(next(iter(agg.values())))} patients) ===")
    rows = []
    res = {}
    for name, n_cls in TASKS:
        r = evaluate(name, n_cls, agg[name])
        res[name] = r
        lo, hi = r["kappa_ci"]
        rows.append({
            "sign": name,
            "ctrl": "NEG" if r["negative_control"] else "",
            "n_pos": r["n_positive"] if r["n_positive"] is not None else "-",
            "kappa": r["kappa"],
            "95% CI": f"{lo}–{hi}",
            "width": r["kappa_ci_width"],
            "auc": r["auc"] if r["auc"] is not None else "-",
        })
    results[h] = {"label": label[h], "n_folds": n_folds, "per_sign": res}
    print(pd.DataFrame(rows).sort_values("kappa", ascending=False).to_string(index=False))

    readable = [v["kappa"] for k, v in res.items() if not v["negative_control"]]
    print(f"\nmedian kappa, 13 readable signs : {np.median(readable):.3f}")
    for c in sorted(NEG_CTRL):
        lo, hi = res[c]["kappa_ci"]
        crosses = lo is not None and lo <= 0 <= hi
        print(f"neg control {c:<26} kappa {res[c]['kappa']:.3f} "
              f"[{lo}, {hi}]   CI includes 0: {crosses}")

if len(hashes) > 1:
    print("\n\n=== configuration comparison (kappa with 95% CI) ===")
    rows = []
    for name, _ in TASKS:
        row = {"sign": name, "ctrl": "NEG" if name in NEG_CTRL else ""}
        for h in hashes:
            r = results[h]["per_sign"][name]
            lo, hi = r["kappa_ci"]
            row[results[h]["label"]] = f"{r['kappa']:.3f} [{lo}, {hi}]"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nOverlapping intervals mean the configurations are not distinguishable "
          "for that sign. Read the overlap, not the point estimates.")

dest = OUT / "bootstrap_ci_p3.json"
dest.write_text(json.dumps(
    {"n_boot": N_BOOT, "seed": SEED,
     "note": "Patient-level bootstrap. Binary thresholds fixed at the full-sample "
             "prevalence-matched value and held constant across resamples.",
     "configs": results}, indent=2, default=str), encoding="utf-8")
print(f"\nsaved -> {dest}")
