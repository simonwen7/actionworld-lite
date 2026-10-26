# M6 — TRAIN-only contact-gated XY residual experiment

## Question and hypothesis

Can a lightweight, capped, contact-conditioned residual fitted on TRAIN episodes
reduce one-step XY motion error beyond the immutable M4 IC-01 predictor?
This is an experiment, not an assumed improvement or a new held-out TEST result.

## Protocol

- Source file SHA-256: `fb7362cb7cacea2c28c771ae64378df09deaf04862de78b72ebf1cadb89be586`
- TRAIN windows: 7938; VALIDATION windows: 1701.
- TRAIN episode-level selection: 30 fit / 12 select per mode (90/36 total).
- Linear ridge regression on predicted-contact windows only; no future state at inference.
- TRAIN-only lambda and correction strength selection. Refit full TRAIN before VAL.
- Selected TRAIN lambda / alpha: 1.0 / 1.0.
- Bounded XY correction: maximum 0.004 meters per step; no correction when predicted no-contact.
- Baselines: frozen IC-01 and TRAIN-mean contact residual; six-feature ablation.
- No checkpoint retraining, TEST access, or validation-dependent method selection.
- Moving cohort: true displacement >= 0.001 m, evaluation grouping only.
- Metrics: mean and 90th-percentile Euclidean XY displacement and next-position errors (cm).

## One-pass VALIDATION results

| Method | Motion mean (cm) | Motion P90 (cm) | Next XY mean (cm) |
|---|---:|---:|---:|
| frozen_ic01 | 0.156625 | 0.651532 | 0.410735 |
| train_constant_residual | 0.157412 | 0.650584 | 0.411385 |
| m6_full_ridge | 0.135983 | 0.568845 | 0.397596 |
| m6_feature_ablation | 0.155883 | 0.640835 | 0.408401 |

| Cohort | Windows | IC-01 motion mean (cm) | M6 motion mean (cm) |
|---|---:|---:|---:|
| overall | 1701 | 0.156625 | 0.135983 |
| moving | 461 | 0.549755 | 0.456303 |
| static | 1240 | 0.010469 | 0.016896 |
| contact | 488 | 0.517752 | 0.442810 |
| no_contact | 1213 | 0.011341 | 0.012544 |
| center_push | 567 | 0.266350 | 0.235725 |
| offset_push | 567 | 0.203523 | 0.172222 |
| free_motion | 567 | 0.000001 | 0.000001 |

- Mean change (M6 minus IC-01): **-0.020642 cm** (negative = improvement).
- Paired 27-episode bootstrap 95% CI: [-0.031218, -0.011330] cm (2,000 draws; descriptive).
- Predeclared conclusion: **Lower validation mean motion error than frozen IC-01.**
- An improvement in overall mean does not imply every cohort, P90, or independent domain improved.

## Provenance and limitations

- Model JSON SHA-256: `7e2bcba8eb9f2ad24690149b590ccfebc12e02ee82ecca72409c45aefcd50f44`
- VAL report JSON SHA-256: `e705812dac089e0c2722d389f37419318b7aab7c46cb7da2fe8ec2823e7f08b9`
- Exact episode filenames and parameters are in model.json and validation_report.json.
- Real-observation-conditioned one-step physical-state prediction only; no next-RGB synthesis.
- Scene-specific PushT colors and fixed MuJoCo physics; no unseen-domain generalization.
- TEST from M4 was already consumed and was NOT opened or rerun in M6.
- VALIDATION is not an independent newly held-out TEST; do not tune based on this report.
- If the method does not outperform baselines, retain the negative result without adjustment.

## Reproduction boundary

Script entry point: `python -B -m src.research.m6_residual_v01 --run-train-val`.
The original experiment is run once; rerunning would require a separately approved protocol
and a new versioned output namespace. Do not rerun M4 TEST.
