# Evaluation and recorded results

This guide separates three evidence populations that **must not be conflated**: M4's one-time held-out **TEST**, M6's one-pass **VALIDATION**, and M8's **Production acceptance** actions. None is a guarantee of performance outside the fixed simulated environment.

## Dataset and split protocol

The main M3 corpus contains **180 episodes**, each with **64 controlled actions**. Splitting is by episode: **126 TRAIN / 27 VALIDATION / 27 TEST**, yielding **7,938 / 1,701 / 1,701** overlapping one-step windows. The three sampling modes are center push, offset push, and free motion. The split and data provenance are documented in [M3 dataset and baselines](../research/M3_DATASET_BASELINES.md).

The system does not randomize object geometry, mass, friction, camera, or other physical domain parameters. The data does not represent a general-purpose physical interaction benchmark.

## M4 — one-time held-out TEST (IC-01)

| Headline metric | Result |
| --- | ---: |
| One-step windows | **1,701** |
| Mean current XY error | **0.263062 cm** |
| Mean next XY error | **0.339025 cm** |
| Mean motion error | **0.168006 cm** |
| Mean next-yaw circular error | **2.478032°** |
| Next-contact precision / recall / F1 | **0.949260 / 0.967672 / 0.958378** |

The four predeclared comparative TEST criteria passed. The original TEST was opened and evaluated **exactly once**; do **not rerun it** to revise README figures. The primary scientific and artifact-integrity evidence is [M4 acceptance record](M4_ACCEPTANCE_RECORD.md). These findings establish one-step prediction on held-out episodes from the **same frozen simulation family**, not unseen-domain generalization.

## M6 — one-pass VALIDATION (contact residual)

| Predictor | Mean XY motion error | Motion-error P90 | Mean next-XY error |
| --- | ---: | ---: | ---: |
| Frozen IC-01 | 0.156625 cm | 0.651532 cm | 0.410735 cm |
| M6 contact-gated ridge | **0.135983 cm** | **0.568845 cm** | **0.397596 cm** |

Mean motion-error difference (M6 − IC-01): **−0.020642 cm**, approximately **13.18% lower**. The descriptive, episode-paired bootstrap 95% interval for the difference was **[−0.031218, −0.011330] cm** with 2,000 resamples across 27 validation episodes.

This was a one-pass **VALIDATION** analysis after TRAIN-only selection and refitting, not a new independent M6 TEST. Results were not used to retune the frozen predictor.

**Negative findings are retained:** on the static cohort, mean motion error worsened from **0.010469 to 0.016896 cm**; on no-contact it worsened from **0.011341 to 0.012544 cm**. See the complete [M6 research record](../research/M6_RESIDUAL_EXPERIMENT.md) for contact/moving cohorts, ablations, baseline comparisons, selection protocol and provenance hashes.

## M8 — live Production acceptance (2026-10-26)

The real IC-01 + M6 + MuJoCo demo ran through a Vercel Container on Hobby. The recorded acceptance results include:

- **479/479 assertions** across 138 general HTTP calls, including reset, action counting, state and error reconciliation, bound checks, tamper rejection and resource path isolation.
- **416/416 assertions** for a separate 14-action contact sequence. Predicted-contact corrections measured **2.48–4.00 mm**, with zero correction on predicted no-contact.
- **1.05 s** median action latency, with recorded cold resets of approximately **6.9–7.6 s**.
- In the recorded live contact sequence, M6's mean motion error **0.744 cm** was worse than IC-01's **0.529 cm**; this does not erase the different, historical VALIDATION result.

These are **assertion counts**, not 895 independent statistical tests. The original acceptance methods and results are in [M8 Step 3 deployment record](M8_STEP3_DEPLOYMENT.md).

## Cross-platform and acceptance qualifications

MuJoCo contact behavior differs between macOS/arm64 and deployed Linux/x86_64; the documented difference reached roughly **0.9 mm** after nine contact steps. Linux versus macOS rendering differences shift a small number of model-input pixels, affecting some predictions. Same-build macOS and Linux/arm64 restore-versus-continuous checks passed, but the equivalent **x86_64 contact parity check was not directly measured**. Direct Vercel memory instrumentation was unavailable. Preview was skipped before first Production deploy.

Therefore **M8 final verdict remains CONDITIONAL**, not full PASS, unless a later dated closure record provides and verifies the missing evidence.
