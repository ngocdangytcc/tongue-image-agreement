"""P3 — acquisition-shortcut audit aimed at the tongue-sign labels (n = 1,220).

    python sign_shortcut_audit.py --work "G:/My Drive/tongue-ai-work" --out audit.json

Question: can the panel's rating of a sign be predicted from how and when the photograph
was taken, without the tongue? Three channels, each a random forest under patient-grouped
5-fold cross-validation (seed 20260915), out-of-fold AUC (one-vs-rest macro AUC for signs
with more than two levels):
  file metadata  - file size and file modification time;
  EXIF           - capture time and camera model;
  image border   - colour mean and SD of a 16-pixel band around the frame (160 x 160 resize),
                   the only channel a deployed model actually receives.
An AUC near 0.5 means the channel carries no information about the label. Patients
photographed during the pilot (first dorsal photograph before 2025-01-01) are excluded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

SEED, FOLDS = 20260915, 5
PILOT_END = pd.Timestamp("2025-01-01")
SIGNS = ["Tongue_Color", "Tongue_Shape", "Tongue_Moisture", "Coating_Color", "Coating_Thickness",
         "Tongue_Swollen", "Coating_Peeled", "Coating_Rotten", "Coating_Greasy", "Tongue_Scalloped",
         "Tongue_Cracked", "Tongue_Prickly", "Tongue_Stasis_Spots"]


def border_feats(p, band=16):
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB").resize((160, 160)), dtype=np.float32)
    b = np.concatenate([a[:band].reshape(-1, 3), a[-band:].reshape(-1, 3),
                        a[:, :band].reshape(-1, 3), a[:, -band:].reshape(-1, 3)])
    return np.concatenate([b.mean(0), b.std(0)])


def grouped_auc(X, y, g):
    y = pd.Series(y).astype("category").cat.codes.to_numpy()
    if len(np.unique(y)) < 2:
        return None
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    p = cross_val_predict(clf, X, y, groups=g, cv=cv, method="predict_proba")
    if p.shape[1] == 2:
        return round(float(roc_auc_score(y, p[:, 1])), 3)
    return round(float(roc_auc_score(y, p, multi_class="ovr", average="macro")), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="G:/My Drive/tongue-ai-work")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = Path(a.work)
    m = pd.read_csv(w / "manifest.csv")
    d = m[(m.kind == "image") & (m.parse_ok == 1) & (m.view == "dorsal")].copy()
    d["patient_id"] = d.patient_id.astype(int)
    d["t"] = pd.to_datetime(d.exif_datetime, format="%Y:%m:%d %H:%M:%S", errors="coerce")
    first = d.groupby("patient_id").t.min()
    pilot = set(first[first < PILOT_END].index)
    d = d[~d.patient_id.isin(pilot)].reset_index(drop=True)
    cohort = pd.read_csv(w / "Data.csv", encoding="utf-8-sig")
    d = d.merge(cohort[["Patient_ID"] + SIGNS], left_on="patient_id", right_on="Patient_ID")
    print(f"{len(d)} photographs, {d.patient_id.nunique()} patients (pilot excluded: {len(pilot)})", flush=True)

    X_meta = d[["bytes", "mtime"]].to_numpy(dtype=float)
    X_exif = np.column_stack([(d.t.astype("int64") / 1e9).where(d.t.notna(), 0).to_numpy(),
                              d.exif_model.astype("category").cat.codes.to_numpy()])
    feats = []
    for i, f in enumerate(d.filename):
        feats.append(border_feats(w / "png_512" / "dorsal" / (Path(f).stem + ".png")))
        if i % 250 == 0:
            print(f"  border features {i}/{len(d)}", flush=True)
    X_border = np.vstack(feats)

    out = {"n_photographs": int(len(d)), "n_patients": int(d.patient_id.nunique()),
           "method": "random forest (300 trees), patient-grouped 5-fold CV, out-of-fold AUC "
                     "(one-vs-rest macro for multi-level signs)", "signs": {}}
    for s in SIGNS:
        ok = d[s].notna().to_numpy()
        y, g = d.loc[ok, s].to_numpy(), d.loc[ok, "patient_id"].to_numpy()
        out["signs"][s] = {"n_photographs": int(ok.sum()),
                           "file_metadata_auc": grouped_auc(X_meta[ok], y, g),
                           "exif_auc": grouped_auc(X_exif[ok], y, g),
                           "image_border_auc": grouped_auc(X_border[ok], y, g)}
        print(s, out["signs"][s], flush=True)
    for ch in ("file_metadata_auc", "exif_auc", "image_border_auc"):
        v = [x[ch] for x in out["signs"].values() if x[ch] is not None]
        out[f"median_{ch}"] = round(float(np.median(v)), 3)
        out[f"max_{ch}"] = round(float(np.max(v)), 3)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("median", "max"))}))
