"""P3 — recompute every per-sign statistic from the archived out-of-fold predictions,
excluding the six patients photographed during the pilot of the photography protocol.

    python recompute_excluding_pilot.py --work "G:/My Drive/tongue-ai-work" --out results_n1220.json

No GPU and no retraining. The trained models are unchanged (author decision, 10 Nov 2026):
the six pilot patients stayed in the training folds, and are removed from evaluation only.
Pilot patients are those whose first dorsal photograph was taken before 1 January 2025
(EXIF; the pilot photographs date from 17-18 December 2024 and the first cohort photograph
from 4 January 2025, so any cut-off between the two selects the same six).

Distended sublingual veins (rated from the sublingual photograph) and tongue movement (rated
from the tongue video) are evaluated only in patients whose sublingual photograph or video
exists; the other 13 signs are rated from the dorsal photographs and use all 1,220.

The script first reproduces the archived full-cohort values (--validate) so that the
computation is shown to be the one that produced the published numbers, then evaluates on
the cohort without the pilot. The decision rule is re-applied on the evaluated sample:
binary signs are thresholded so that predicted prevalence equals observed prevalence.
Bootstrap: 1,000 patient-level resamples, seed 20260915, thresholds held fixed; kappa,
AUC, sensitivity and specificity share one random stream and Gwet's AC1 a second stream
with the same seed, as in run_final.py and add_gwet_ac1.py.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)

N_BOOT, SEED, FOLDS = 1000, 20260915, 5
PILOT_END = pd.Timestamp("2025-01-01")
CONFIGS = {"6ep": "5a2b7673b283", "12ep": "b1ba0615f548"}

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


def gwet_ac1(y1, y2, q):
    n = len(y1)
    pa = float((y1 == y2).mean())
    pe = sum((((y1 == k).sum() + (y2 == k).sum()) / (2.0 * n)) *
             (1 - ((y1 == k).sum() + (y2 == k).sum()) / (2.0 * n)) for k in range(q)) / (q - 1)
    return ((pa - pe) / (1.0 - pe) if pe < 1 else float("nan")), pa


def ci(v):
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else [None, None]


def pilot_ids(work):
    m = pd.read_csv(work / "manifest.csv")
    d = m[(m.kind == "image") & (m.parse_ok == 1) & (m.view == "dorsal")].copy()
    d["t"] = pd.to_datetime(d.exif_datetime, format="%Y:%m:%d %H:%M:%S", errors="coerce")
    first = d.groupby("patient_id").t.min()
    return sorted(int(p) for p in first[first < PILOT_END].index)


def load_oof(fold_dir, h):
    pids, per = [], [[] for _ in TASKS]
    for k in range(FOLDS):
        z = np.load(fold_dir / f"fold{k}_{h}.npz")
        pids.append(z["pid"])
        for j in range(len(TASKS)):
            per[j].append(z[f"t{j}"])
    return np.concatenate(pids), [np.concatenate(p) for p in per]


def auc_of(y, P, n_cls):
    if n_cls == 1:
        return roc_auc_score(y, P[:, 0])
    if n_cls == 2:
        return roc_auc_score(y, P[:, 1])
    return roc_auc_score(y, P, multi_class="ovr", average="macro")


def evaluate_sign(name, n_cls, pid, p, truth, keep):
    df = pd.DataFrame({"pid": pid})
    if p.ndim == 1:
        df["p0"] = p
    else:
        for i in range(p.shape[1]):
            df[f"p{i}"] = p[:, i]
    agg = df.groupby("pid").mean()                   # two dorsal photographs averaged
    agg = agg[agg.index.isin(keep)]
    if name in CATEGORICAL:
        y = truth.loc[agg.index, name].map({v: i for i, v in enumerate(CATEGORICAL[name])})
        q = len(CATEGORICAL[name])
    else:
        y = truth.loc[agg.index, name]
        q = 2
    ok = y.notna().to_numpy()
    yv = y[ok].astype(int).to_numpy()
    P = agg.to_numpy()[ok]
    out = {"n": int(ok.sum()), "kind": "binary" if n_cls == 1 else "multiclass",
           "negative_control": name in NEG_CTRL}
    if n_cls == 1:
        thr = float(np.quantile(P[:, 0], 1 - yv.mean()))
        yh = (P[:, 0] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, yh, labels=[0, 1]).ravel()
        out.update(prevalence=round(float(yv.mean()), 4), n_positive=int(yv.sum()),
                   sensitivity=round(float(recall_score(yv, yh, zero_division=0)), 4),
                   specificity=round(float(recall_score(1 - yv, 1 - yh, zero_division=0)), 4),
                   two_by_two={"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
    else:
        yh = P.argmax(1)
        out.update(n_classes=n_cls,
                   majority_baseline=round(float(pd.Series(yv).value_counts(normalize=True).max()), 4),
                   accuracy=round(float(accuracy_score(yv, yh)), 4),
                   macro_f1=round(float(f1_score(yv, yh, average="macro", zero_division=0)), 4),
                   confusion_matrix={"labels": CATEGORICAL[name],
                                     "rows_panel_cols_model": confusion_matrix(yv, yh, labels=list(range(n_cls))).tolist()})
    out["kappa"] = round(float(cohen_kappa_score(yv, yh)), 4)
    out["auc"] = round(float(auc_of(yv, P, n_cls)), 4)
    ac, pa = gwet_ac1(yv, yh, q)
    out["ac1"], out["percent_agreement"] = round(float(ac), 4), round(pa * 100, 1)
    if name == "Coating_Thickness" or n_cls == 2:
        out["sensitivity_thick"] = round(float(recall_score(yv, yh, zero_division=0)), 4)
        out["specificity_thick"] = round(float(recall_score(1 - yv, 1 - yh, zero_division=0)), 4)
    if n_cls > 2:
        out["kappa_linear_weighted"] = round(float(cohen_kappa_score(yv, yh, weights="linear")), 4)

    rng = np.random.default_rng(SEED)                 # stream 1: kappa, AUC, Se, Sp
    ks, au, se, sp = [], [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yv), len(yv))
        yb = yv[idx]
        if len(np.unique(yb)) < 2:
            continue
        ks.append(cohen_kappa_score(yb, yh[idx]))
        try:
            au.append(auc_of(yb, P[idx], n_cls))
        except ValueError:
            pass
        if n_cls == 1:
            se.append(recall_score(yb, yh[idx], zero_division=0))
            sp.append(recall_score(1 - yb, 1 - yh[idx], zero_division=0))
    rng = np.random.default_rng(SEED)                 # stream 2: AC1
    acs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yv), len(yv))
        yb, hb = yv[idx], yh[idx]
        if len(np.unique(yb)) < 2:
            continue
        a, _ = gwet_ac1(yb, hb, q)
        if not np.isnan(a):
            acs.append(a)
    out.update(kappa_ci=ci(ks), auc_ci=ci(au), ac1_ci=ci(acs), n_valid_resamples=len(ks))
    if n_cls == 1:
        out.update(sensitivity_ci=ci(se), specificity_ci=ci(sp))
    return out


def missing_source_ids(work):
    """Patients whose reference rating had no source file for a control sign.

    The panel rated distended sublingual veins from the sublingual photograph and tongue
    movement from the tongue video. A patient with no such file is excluded from that
    sign only (author decision, 11 Nov 2026). Returns {sign: set of patient ids}.
    """
    m = pd.read_csv(work / "manifest.csv")
    dorsal = set(m[(m.kind == "image") & (m.view == "dorsal")].patient_id.astype(int))
    sub = set(m[(m.kind == "image") & (m.view == "sublingual")].patient_id.astype(int))
    vid = set(m[m.kind == "video"].patient_id.astype(int))
    return {"Tongue_Sublingual_Veins": dorsal - sub, "Tongue_Movement": dorsal - vid}


def run(work, keep, label, sign_exclusions=None):
    truth = pd.read_csv(work / "Data.csv", encoding="utf-8-sig").set_index("Patient_ID")
    sign_exclusions = sign_exclusions or {}
    res = {"label": label, "patients_evaluated": len(keep), "configs": {},
           "sign_specific_exclusions": {k: len(v & keep) for k, v in sign_exclusions.items()}}
    for cfg, h in CONFIGS.items():
        pid, per = load_oof(work / "p3_final" / "folds", h)
        signs = {name: evaluate_sign(name, n_cls, pid, per[j], truth, keep - sign_exclusions.get(name, set()))
                 for j, (name, n_cls) in enumerate(TASKS)}
        readable = [v for k, v in signs.items() if not v["negative_control"]]
        res["configs"][cfg] = {"config_hash": h, "per_sign": signs,
                               "median_kappa_readable": round(float(np.median([v["kappa"] for v in readable])), 4),
                               "median_ac1_readable": round(float(np.median([v["ac1"] for v in readable])), 4)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="G:/My Drive/tongue-ai-work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    pilot = pilot_ids(work)
    assert len(pilot) == 6, f"expected 6 pilot patients, found {len(pilot)}"
    all_ids = set(pd.read_csv(work / "Data.csv", encoding="utf-8-sig").Patient_ID.astype(int))
    report = {"pilot_patients_excluded": len(pilot),
              "rule": "first dorsal photograph (EXIF) before 2025-01-01",
              "full_cohort": run(work, all_ids, "all patients (validation against the archived run)"),
              "excluding_pilot": run(work, all_ids - set(pilot), "pilot excluded from evaluation; control signs "
                                     "evaluated only where their rating source exists", missing_source_ids(work))}
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")   # no patient IDs written
    for part in ("full_cohort", "excluding_pilot"):
        c = report[part]["configs"]
        print(part, report[part]["patients_evaluated"],
              {k: (v["median_kappa_readable"], v["median_ac1_readable"]) for k, v in c.items()})
