# Tongue-image agreement with an expert panel

Code and aggregate results for a cross-sectional study. The study measures how well a
convolutional network reproduces a five-physician traditional-medicine panel's ratings of
15 tongue signs from smartphone photographs. The cohort is 1,220 adults scheduled for upper
endoscopy at University Medical Center Ho Chi Minh City, Vietnam.

The repository lets readers inspect and re-run the analysis behind the manuscript.
**It contains no patient data, no photographs and no patient identifiers.** See
[Data availability](#data-availability).

## What the study measures

The estimand is **agreement with a human reference standard**, not diagnostic accuracy.
- Reference standard: the majority vote of five traditional-medicine physicians (mean experience
  10.8 years, range 7 to 14), blinded to the endoscopic findings.
  - 13 signs were rated from the dorsal photographs.
  - Distended sublingual veins were rated from the sublingual photograph.
  - Tongue movement was rated from the tongue video.
- Model input: the two dorsal photographs of each patient only. Their scores are averaged per patient.
- Statistics: Cohen's kappa (primary) and Gwet's AC1, with patient-level bootstrap 95% confidence intervals.
- Negative controls: tongue movement and distended sublingual veins. A dorsal still photograph
  cannot show either, and both were specified before training.

Atrophic gastritis is **not** an endpoint here. It is used only to stratify the
cross-validation folds. Two other manuscripts from the same cohort have different estimands
(see [Related work](#related-work)).

## Cohort and the six pilot patients

- 1,226 patients were photographed.
- Six were photographed on 17–18 December 2024, during a pilot of the photography protocol before
  formal recruitment. They are **excluded from evaluation**, leaving 1,220 patients and 2,440 dorsal
  photographs.
- The models were trained once, on all 1,226 patients, under patient-grouped 5-fold cross-validation,
  and were not retrained. The six pilot patients' photographs therefore sit in the training folds of
  the other four folds.
- The two control signs are analysed only where their rating source exists. Distended sublingual
  veins are analysed in patients with a sublingual photograph, and tongue movement in patients with
  a tongue video: 1,217 each.
- `analysis/recompute_excluding_pilot.py` recomputes every statistic from the archived
  out-of-fold predictions. It first reproduces the archived full-cohort run (207 of 207 values
  identical), then evaluates on the 1,220.

## Headline numbers (n = 1,220; 6-epoch configuration)

| | |
|---|---|
| Median kappa, 13 signs visible in a dorsal photograph | 0.324 |
| Median Gwet's AC1, same signs | 0.572 |
| Cracked tongue | kappa 0.643 (95% CI 0.602 to 0.685) |
| Negative control: tongue movement (n = 1,217) | kappa 0.011 (95% CI −0.017 to 0.039); AC1 0.639 |
| Published median inter-practitioner kappa | 0.34 (Jacobson et al. 2019, 21 studies) |

`results/results.json` holds every number, with its confidence interval.

## Acquisition-shortcut audit (aimed at the sign labels)

Can the panel's rating be predicted from how and when the photograph was taken, without the
tongue? Each channel was tested with a random forest under patient-grouped 5-fold
cross-validation, with out-of-fold AUC reported for each of the 13 visible signs
(`analysis/sign_shortcut_audit.py`).

| Channel | Median AUC | Maximum AUC |
|---|---|---|
| File metadata (size, modification time) | 0.537 | 0.562 |
| EXIF (capture time, camera model) | 0.532 | 0.571 |
| Image border (16-pixel band colour) | 0.518 | 0.586 |

All three channels are close to chance. Of the three, only the image border is an input that
a deployed model receives.

## Layout

```
analysis/
  run_final.py                   training and evaluation run (Colab, GPU)
  bootstrap_ci.py                patient-level bootstrap intervals (archived run)
  dump_p3_splits.py              reconstructs the training fold assignment (all 1,226) and checks it is leakage-free
  recompute_excluding_pilot.py   n = 1,220 evaluation from the archived out-of-fold predictions
  sign_shortcut_audit.py         acquisition-shortcut audit aimed at the sign labels
  01_inventory_qc.ipynb          inventory and quality control (outputs stripped)
results/results.json             public results file (File S1 of the manuscript)
```

## Reproducing

Training needs a GPU and the photographs, neither of which is in this repository.
`run_final.py` ran on Google Colab Pro (NVIDIA L4) with python 3.13.15, torch 2.11.0+cu128,
timm 1.0.29, scikit-learn 1.6.1, numpy 2.1.3 and pandas 2.2.3.

The n = 1,220 evaluation and the shortcut audit run on a CPU. They need the archived
out-of-fold predictions, `Data.csv` and the manifest, which are available on request (below).
They were run with python 3.12.10, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.9.0 and scipy 1.14.1:

```bash
python analysis/recompute_excluding_pilot.py --work <data folder> --out recompute.json
python analysis/sign_shortcut_audit.py --work <data folder> --out audit.json
```

Bootstrap: 1,000 patient-level resamples, seed 20260915. Binary signs are thresholded so that
predicted prevalence equals observed prevalence. The threshold is held fixed across resamples.

## Data availability

The photographs and the clinical table are not in this repository. They show the lower face
and were collected under an ethics approval that does not cover public release. De-identified
data, the photographs and the archived predictions are available from the corresponding author
on reasonable request, subject to the approval of the ethics board.

Ethics approval: Board of Ethics in Biomedical Research, University of Medicine and Pharmacy at
Ho Chi Minh City, Decision No. 3911/ĐHYD-HĐĐĐ, 2 December 2024.

## Related work

Two manuscripts from the same cohort have different estimands:
- a prediction model of endoscopic atrophic gastritis from structured pre-endoscopic features,
  which does not use the photographs;
- an analysis of the same dorsal photographs as a predictor of endoscopic atrophy (under review).

## Citation

See `CITATION.cff`. Please cite the paper rather than this repository.

## Licence

The code is under the MIT licence (`LICENSE`). `results/results.json` and the text of this README
are under CC BY 4.0.
