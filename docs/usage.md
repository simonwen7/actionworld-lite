# Usage and reproduction boundary

## Use the public interactive demo

Visit **<https://actionworld-lite.vercel.app>**. Choose a 5, 10, or 20 mm step size and one direction (up/down/left/right). The page displays the latest real MuJoCo image, separate XY state comparison, contact values, and error metrics. Reset creates a new scene and scored-action count zero.

A cold reset can be slower than ordinary actions. The session lasts **15 minutes** from initialization or up to **64 scored actions**, whichever ends first. Sessions are browser-carried and signed, not stored in a server-side database. The service runs on a limited free Hobby allowance and availability is not guaranteed.

## What a fresh clone includes

The public repository contains source for simulation, dataset generation, IC-01, M6 residual logic, GUI/web demos, testing and deployment staging. It **does not include**:

- the two frozen `.pt` weight checkpoints;
- the fitted M6 `model.json` runtime artifact;
- the generated training, validation and TEST data archives;
- a pre-approved redistribution/license package for the private runtime files.

A fresh clone cannot reproduce the trained-model demo without separately authorized copies of those artifacts. **Do not download, fabricate, regenerate, replace, or publicly redistribute weights as an undocumented workaround.** See [models](models.md).

## Source-only local checks

These examples assume you already have a compatible **Python 3.12** environment with the declared dependencies; they do not create or fetch missing frozen model artifacts. Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.test_m8_synthetic -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.test_m8_api_synthetic -v
```

The first suite is synthetic. The mocked HTTP suite additionally requires the local FastAPI and `httpx` test dependencies. If those dependencies are missing, do not claim the test ran. A complete reproducible fresh-machine install is **not** yet documented or proven with a transitive lockfile.

## Local physical demonstration (existing accepted machine)

The [M7 local release record](M7_LOCAL_RELEASE.md) provides these commands for a previously configured macOS environment holding the accepted frozen runtime inputs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -B -m src.demo.m7_release --self-test
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -B -m src.demo.m7_release --run-demo
```

The self-test does **not** open private datasets or launch the GUI; the demo needs a graphical desktop and the exact accepted checkpoints and fitted JSON. The commands are not a promise that an arbitrary fresh clone will run.

## Research evaluation restrictions

Do not rerun the sealed **M4 one-time TEST** or use its results for new model selection. Do not silently rerun the original **M6 experiment** under the same namespace. Results should be cited from the existing records: [M4 acceptance](M4_ACCEPTANCE_RECORD.md), [M6 experiment](../research/M6_RESIDUAL_EXPERIMENT.md), and [M8 deployment](M8_STEP3_DEPLOYMENT.md).
