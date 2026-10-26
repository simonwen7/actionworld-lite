# ActionWorld Lite — M7 Local Research Demo Release

**Release candidate:** M7 local-only integration, **not a fresh-clone public runnable release**.  
**Platform:** macOS graphical desktop; existing Python 3.12.14 environment, NumPy 2.5.4, PyTorch 2.14.1, MuJoCo 3.17.0, and Tkinter.  
**Scope:** fixed PushT-style simulation; one-step model comparison conditioned on the last two *real* rendered images and one user-selected action. No future RGB generation, no autonomous learned rollout.

## Run locally

From the repository root, with the previously installed virtual environment, both approved M4 checkpoints, and the accepted M6 JSON artifacts already on this Mac:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -B -m src.demo.m7_release --self-test
```

The command above uses synthetic arrays only and **does not launch a GUI, load checkpoints or open episode data**.

A subsequent, separately authorized final acceptance may run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -B -m src.demo.m7_release --run-demo
```

`--run-demo` is a macOS **local physical simulator** and requires a graphical desktop. A fresh clone currently does **not** contain all required files. The program fails closed if hashes or the fitted-model schema do not match; it never trains or downloads a replacement.

## Runtime frozen inputs

| Path | SHA-256 | Availability |
| --- | --- | --- |
| `src/models/integrated_pusht_m4.py` | `e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a` | Tracked |
| `src/demo/m5_interactive.py` | `5481176dabf76f28c2ad4411eadc698e81a4506af07d2b78abc5fc7c89fd0dac` | Tracked |
| `src/research/m6_residual_v01.py` | `fb7362cb7cacea2c28c771ae64378df09deaf04862de78b72ebf1cadb89be586` | Tracked |
| `data/processed/m4_visual_v03/best_model.pt` | `d0e24224680e557b2619595a3adcfc09183d38bc43a22cb699a074df5f8a95bd` | Local / ignored |
| `data/processed/m4_yaw_v04/best_model.pt` | `50cb95f37ed1b2a342e4ef669324518d45b5f6509be2fa53088f4ecf8a92d93e` | Local / ignored |
| `data/processed/m6_residual_v01/model.json` | `7e2bcba8eb9f2ad24690149b590ccfebc12e02ee82ecca72409c45aefcd50f44` | Local / ignored at M7 CHECK |
| `data/processed/m6_residual_v01/validation_report.json` | `e705812dac089e0c2722d389f37419318b7aab7c46cb7da2fe8ec2823e7f08b9` | Local / ignored at M7 CHECK |

The original M4 TEST archive and its eight checksummed records must remain untouched. The final demo does not require TEST data, TRAIN/VALIDATION episode arrays or training scripts.

## What you see

The window displays the actual MuJoCo image, an orange physical-state outline, a **blue dashed** frozen IC-01 next-state prediction and a **purple dotted** M6 next-state prediction in XY world coordinates. Yaw and predicted contact are inherited unchanged from IC-01; M6 corrects **only XY motion**. All predictions precede a single matching physics action. The real *next* MuJoCo frame is rendered by MuJoCo, not synthesized by the AI.

Click 5, 10 or 20 mm and a direction. The initial 0 mm physics step used to create two real frames is excluded from scored actions. A deliberately pressed Zero button is scored, like any other user action. Reset returns the scored count to zero. For auditability, the demo records reset, action, error and close events as append-only `.jsonl` files under `~/Desktop/ActionWorldLite_M7_Sessions/` **outside the repository**. Each action record includes actual and both predicted XY values, their centimeter errors, predicted/actual contact and exactly one inference/physics-step count. Screenshot evidence can supplement, not replace, these logs.

Action callbacks perform inference synchronously on Tk's event loop. Buttons intentionally avoid keyboard auto-repeat. Any action failure disables further action buttons until Reset; no automatic retry occurs. If the window is closed, MuJoCo resources and the session log are released. A clean process exit alone is not the complete release acceptance: Step 3 must inspect actual GUI interaction, original logs and frozen hashes.

## Scientific claim boundary

M6 TRAIN-only fitted ridge residual, evaluated **once on VALIDATION** (1,701 windows, 27 episodes):

- Frozen IC-01 mean XY motion error: **0.156625 cm**.
- M6 full ridge mean XY motion error: **0.135983 cm** (~**13.18%** lower).
- Episode-paired bootstrap 95% descriptive interval for the M6 minus IC-01 change: **[-0.031218, -0.011330] cm**.
- **Static** mean motion error increased from **0.010469** to **0.016896 cm**; **no-contact** from **0.011341** to **0.012544 cm**.
- The contact-gated M6 residual is capped at **0.004 m (4 mm)**. Predicted no-contact receives zero correction.

These are scene-specific VALIDATION observations, **not an independent new TEST or evidence of generalization**. The original M4 TEST was consumed exactly once, and its accepted results have not changed. The M5 exact ten-click transcript was **waived by the user**, not retrospectively shown to have passed.

## Release limitations and publication blockers

1. **Fresh-clone gap:** M4 PyTorch checkpoints are ignored by Git, with no approved public distribution mechanism. At M7 CHECK, the two M6 JSON artifacts are also ignored and absent from the repository commit. Their locally accepted hashes are not substitutes for shipping them. Do **not** claim public plug-and-play reproducibility yet.
2. **Dependencies:** `requirements-m3.txt` pins MuJoCo and NumPy but not Torch. No transitive lockfile or fresh-machine installation audit currently exists. Tkinter and macOS GUI support depend on the chosen Python distribution.
3. **License:** no repository `LICENSE` was present at M7 CHECK. Do not assert a specific project license or checkpoint redistribution permission until the owner decides and third-party terms are reviewed.
4. **Scope:** fixed geometry, color-prior observations, constrained pusher actions; no transfer guarantee, no physical robot and no free-running RGB prediction.
5. **Quality:** numerical validation improvement coexists with static/no-contact regression. Individual live actions may show worse errors than the baseline.
6. **Historical documentation:** README/research setup status and the immutable M4 acceptance record have stale milestone-status language; do not alter frozen M4 evidence to hide history. A future owner-controlled documentation release may clarify these status lines.

**Distribution decision and Git actions are reserved to the repository owner.** Do not automatically commit ignored JSON, binary checkpoints or choose a license. M7 is initially a verifiable *local* release candidate on the existing machine, pending Step 3 final acceptance.
