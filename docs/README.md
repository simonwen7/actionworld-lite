# ActionWorld Lite — Documentation

This index separates **current reader guides** from **historical, milestone-scoped evidence**. A milestone document's earlier status language is a snapshot of that milestone, not the current project status. The current demo is deployed and M8 acceptance is **CONDITIONAL** pending the documented remaining checks.

## Start here

| Guide | Purpose |
| --- | --- |
| [Project README](../README.md) | Short technical overview and live demo |
| [Architecture](architecture.md) | Components, data flow, signed sessions and trust boundaries |
| [Models](models.md) | IC-01 and M6 inputs, outputs and model-specific constraints |
| [Evaluation](evaluation.md) | Distinct TEST, VALIDATION, and Production evidence |
| [Usage](usage.md) | Public demo and safe local workflows |
| [API](api.md) | HTTP request contracts, response shape and errors |
| [Deployment](deployment.md) | Actual Vercel Container architecture, staging and operational constraints |
| [Limitations](limitations.md) | Scientific and deployment limitations; outstanding acceptance gates |
| [Repository map](project-structure.md) | Which directories contain source, evidence and excluded runtime artifacts |

## Primary research documents

| Original record | Evidence |
| --- | --- |
| [M3 dataset and baselines](../research/M3_DATASET_BASELINES.md) | Data generation, split policy, baseline results and hashes |
| [M4 one-time TEST acceptance](M4_ACCEPTANCE_RECORD.md) | Frozen IC-01 evaluation, 1,701 TEST windows and archive integrity |
| [M6 residual experiment](../research/M6_RESIDUAL_EXPERIMENT.md) | TRAIN-only selection, one-pass VALIDATION, bootstrap interval and negative cohorts |
| [Research index](../research/README.md) | Research lineage and reproduction restrictions |

## Implementation and release records

| Original record | Scope |
| --- | --- |
| [M7 local demo release](M7_LOCAL_RELEASE.md) | macOS-only local demonstration and frozen file manifest, as of M7 |
| [M8 Step 2 handoff](M8_WEB_HANDOFF.md) | Web implementation plan and pre-deployment status **at Step 2** |
| [M8 Step 3 deployment record](M8_STEP3_DEPLOYMENT.md) | Actual Container deployment, Production checks, measured differences and gaps |

**Evidence policy:** Keep the historical documents byte-identical; add new clarifications in reader guides or a new dated closure record. Do not retune models using original held-out TEST data. Distinguish a passed acceptance assertion from an independently sampled test case.
