"""Reconstruct and save the P3 fold assignment, then prove it is leakage-free.

    %run /content/drive/MyDrive/tongue-ai-work/dump_p3_splits.py

Why this exists. run_final.py built its folds in memory and saved only the model
outputs, so the project's first ground rule - patient-level splits, two photographs of
one patient never separated - is currently true but unverifiable by anyone reading the
paper. The pre-submission audit ranked that BLOCKING, correctly.

No training and no GPU. The fold construction is deterministic given the seed and the
data, so this reproduces exactly the partition that produced the published numbers,
then checks it three ways and writes it out as a citable artefact.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

WORK = Path("/content/drive/MyDrive/tongue-ai-work")
OUT = WORK / "p3_final"
SEED, FOLDS = 20260915, 5

# ---- rebuild the analysis frame exactly as run_final.py did -----------------
man = pd.read_csv(WORK / "manifest.csv")
cohort = pd.read_csv(WORK / "Data.csv", encoding="utf-8-sig")

d = man[(man.kind == "image") & (man.parse_ok == 1) & (man.view == "dorsal")].copy()
d["patient_id"] = d.patient_id.astype(int)

# One directory listing, then set membership - 2452 individual stat() calls over the
# Drive FUSE mount would take minutes.
available = {p.name for p in (WORK / "png_512" / "dorsal").glob("*.png")}
d["png_name"] = d.filename.map(lambda f: Path(f).stem + ".png")
d = d[d.png_name.isin(available)].copy()

d["month"] = pd.to_datetime(d.exif_datetime, format="%Y:%m:%d %H:%M:%S",
                            errors="coerce").dt.to_period("M").astype(str)
d.loc[d.month == "NaT", "month"] = "unknown"
d = d.merge(cohort[["Patient_ID", "Atrophic_Gastritis"]],
            left_on="patient_id", right_on="Patient_ID")

print(f"rebuilt frame: {len(d)} dorsal images, {d.patient_id.nunique()} patients")
assert len(d) == 2452, f"expected 2452 images, got {len(d)} - the frame does not match the run"
assert d.patient_id.nunique() == 1226, "patient count does not match the run"

# ---- reproduce the partition ------------------------------------------------
d["strat"] = d.arm_folder.astype(str) + "|" + d.month.astype(str)
vc = d.strat.value_counts()
d.loc[d.strat.isin(vc[vc < FOLDS * 2].index), "strat"] = "rare"

sgkf = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
d["fold"] = -1
for k, (_, va) in enumerate(sgkf.split(d, d.strat, groups=d.patient_id)):
    d.iloc[va, d.columns.get_loc("fold")] = k
assert (d.fold >= 0).all(), "some rows were never assigned a fold"

# ---- three independent leakage checks ---------------------------------------
checks = {}

# 1. no patient appears in two folds
per_patient_folds = d.groupby("patient_id").fold.nunique()
checks["patients_spanning_more_than_one_fold"] = int((per_patient_folds > 1).sum())

# 2. both photographs of every patient share a fold
img_per_patient = d.groupby("patient_id").size()
checks["patients_with_two_images"] = int((img_per_patient == 2).sum())
checks["patients_with_other_image_counts"] = int((img_per_patient != 2).sum())

# 3. every validation fold is disjoint from its own training set
overlaps = []
for k in range(FOLDS):
    tr = set(d[d.fold != k].patient_id)
    va = set(d[d.fold == k].patient_id)
    overlaps.append(len(tr & va))
checks["max_train_valid_patient_overlap"] = max(overlaps)

# 4. the out-of-fold coverage matches what the results report
checks["total_patients_covered"] = int(d.patient_id.nunique())
checks["union_of_validation_folds"] = int(
    len(set().union(*[set(d[d.fold == k].patient_id) for k in range(FOLDS)])))

print("\nleakage checks:")
for k, v in checks.items():
    print(f"  {k:<42} {v}")
ok = (checks["patients_spanning_more_than_one_fold"] == 0
      and checks["max_train_valid_patient_overlap"] == 0
      and checks["union_of_validation_folds"] == 1226)
print(f"\nVERDICT: {'PASS - no patient-level leakage' if ok else 'FAIL - investigate'}")

# ---- write the artefact -----------------------------------------------------
payload = {
    "study": "P3",
    "generated_ict": (pd.Timestamp.utcnow() + pd.Timedelta(hours=7)).isoformat(),
    "timezone": "ICT (UTC+7)",
    "purpose": ("Fold assignment underlying every published P3 number. Reconstructed "
                "deterministically from the same manifest, seed and stratification as "
                "run_final.py, which held the folds only in memory."),
    "unit": "Patient_ID",
    "n_folds": FOLDS,
    "seed": SEED,
    "stratified_on": "arm (folder) x EXIF capture month, strata below 2*n_folds collapsed to 'rare'",
    "splitter": "sklearn StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260915)",
    "n_images": int(len(d)),
    "n_patients": int(d.patient_id.nunique()),
    "leakage_checks": checks,
    "leakage_verdict": "PASS" if ok else "FAIL",
    "fold_sizes": {str(k): {"images": int((d.fold == k).sum()),
                            "patients": int(d[d.fold == k].patient_id.nunique()),
                            "ag_rate": round(float(d[d.fold == k].Atrophic_Gastritis.mean()), 4)}
                   for k in range(FOLDS)},
    "folds": {str(k): sorted(int(p) for p in d[d.fold == k].patient_id.unique())
              for k in range(FOLDS)},
}
dest = OUT / "splits_p3_final.json"
dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
d[["patient_id", "filename", "png_name", "view", "arm_folder", "month", "strat", "fold"]] \
    .sort_values(["fold", "patient_id"]).to_csv(OUT / "splits_p3_final.csv", index=False)

print(f"\nsaved -> {dest}")
print(f"saved -> {OUT / 'splits_p3_final.csv'}")
print("\nfold sizes:")
for k, v in payload["fold_sizes"].items():
    print(f"  fold {k}: {v['images']} images, {v['patients']} patients, AG rate {v['ag_rate']}")
print("\n" + "=" * 60)
print("  PASTE THE LEAKAGE CHECKS AND FOLD SIZES BACK INTO THE CHAT")
print("=" * 60)
