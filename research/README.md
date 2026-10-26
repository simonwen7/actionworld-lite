# ActionWorld Lite — Research index

This directory preserves the project's scientific methods, measurement scope and original experiment records. The current system contains a trained, frozen **IC-01** one-step visual predictor, a TRAIN-fitted **M6** XY residual, and a live MuJoCo comparison demo. This index replaces only the old planning-era status summary; the underlying research and acceptance evidence is not rewritten.

## Experiments and accepted records

| Phase | Original source of truth | What it establishes |
| --- | --- | --- |
| M3 dataset and baselines | [M3_DATASET_BASELINES.md](M3_DATASET_BASELINES.md) | 180 episodes, episode-wise TRAIN/VALIDATION/TEST split, data schema, persistence and privileged-state constant-velocity baselines |
| M4 integrated model | [M4 acceptance](../docs/M4_ACCEPTANCE_RECORD.md) | Frozen IC-01, **one-time held-out TEST** metrics, gates, immutable archive hashes |
| M6 contact residual | [M6_RESIDUAL_EXPERIMENT.md](M6_RESIDUAL_EXPERIMENT.md) | TRAIN-only residual selection, one-pass **VALIDATION**, ablations and negative subsets |
| M7 local demo | [M7 release](../docs/M7_LOCAL_RELEASE.md) | Local visual comparison, physical reference and checkpoint distribution restrictions |
| M8 web deployment | [M8 production record](../docs/M8_STEP3_DEPLOYMENT.md) | Real Production online inference, physics, HTTP/browser checks, platform differences and remaining acceptance gates |

For a readable synthesis, see [Evaluation](../docs/evaluation.md), [Models](../docs/models.md), and [Limitations](../docs/limitations.md).

## Evidence rules

- Never reopen or rerun the sealed **M4 TEST** to produce revised headline claims.
- Distinguish **M4 TEST** from **M6 VALIDATION**. The reported **13.18%** reduction is a validation result, not an independent M6 TEST result.
- Keep the reported regressions in static/no-contact subsets and the observed M8 live contact run.
- Preserve frozen sources, training/evaluation provenance, data split policy and checksum evidence.
- Keep experimental or deployment uncertainties visible; a planned or inferred check is not a measured PASS.
- Do not claim public end-to-end reproducibility when checkpoints, fitted runtime JSON and generated datasets are not distributed.

The research records are historical evidence. A reader-facing guide may summarize them, but must link back to the original documents without altering their meaning.
