# Models and prediction contract

ActionWorld Lite has two sequential predictors: **IC-01**, the frozen action-conditioned visual model, and **M6**, a lightweight contact-gated residual for predicted XY displacement. Neither predictor generates RGB pixels.

## IC-01: integrated visual physical-state predictor

Main source: [`src/models/integrated_pusht_m4.py`](../src/models/integrated_pusht_m4.py). It combines:

- **V0.3 visual dynamics**, 424,345 parameters, for action-conditioned motion estimates;
- **V0.4 silhouette yaw specialist**, 231,362 parameters, for the current object yaw;
- fixed TRAIN-fitted visual-to-world calibration for current object and pusher positions;
- a fixed endpoint clearance proxy and TRAIN-selected threshold for next-step contact;
- a **hard XY translation gate** when predicted contact is false. The yaw increment is **not** hard contact-gated by that decision.

Input is `[B, 6, 64, 64]` float32 image data from two consecutive genuine RGB frames, normalized to `[0,1]`, and a `[B,2]` action tensor in units of 0.02 m. Runtime inference is CPU-only and does not use future simulator state or labels.

Output contains estimates of the current XY position and yaw, the next XY position and yaw, predicted displacement, contact prediction and related contact score. The reference MuJoCo state is used **after** the action for evaluation and display, not as the predictor input.

## M6: TRAIN-fitted contact-gated residual

Source: [`src/research/m6_residual_v01.py`](../src/research/m6_residual_v01.py); research record: [M6 residual experiment](../research/M6_RESIDUAL_EXPERIMENT.md).

M6 fits a small ridge residual on predicted-contact TRAIN windows and adds the bounded correction to the **IC-01 XY displacement**. It does not change IC-01's yaw prediction or contact decision.

- Correction bound: **0.004 m (4 mm)** per action.
- Predicted no-contact: **exactly zero** residual correction.
- Selected TRAIN ridge regularization and correction strength: **1.0 / 1.0**.
- TRAIN selection and fitting do not use M4's held-out TEST; M6 was evaluated once on a separate **VALIDATION** protocol.

The measured 13.18% overall motion-error improvement is specific to this validation setup. Several subsets and individual live runs regress; see [Evaluation](evaluation.md).

## Frozen inputs and public distribution

| Runtime file | Repository availability |
| --- | --- |
| `src/models/integrated_pusht_m4.py` | Public tracked source |
| `src/research/m6_residual_v01.py` | Public tracked source |
| `data/processed/m4_visual_v03/best_model.pt` | Private to approved runtime, Git-ignored |
| `data/processed/m4_yaw_v04/best_model.pt` | Private to approved runtime, Git-ignored |
| `data/processed/m6_residual_v01/model.json` | Private to approved runtime, Git-ignored |

The runtime checks SHA-256 hashes and refuses mismatched inputs. The exact frozen manifest and evidence hashes are documented in [M7 release](M7_LOCAL_RELEASE.md) and [M8 deployment](M8_STEP3_DEPLOYMENT.md). Model data is intentionally not provided as public downloads; a checkout of source alone cannot execute the complete predictor.

## Scientific boundary

This is a **hybrid, scene-specific** visual/neural/geometric model on a simplified fixed PushT environment. It is not a universally trained physical foundation model, an embodied robotics policy, a general unseen-scene predictor, or a model trained to synthesize future images.
