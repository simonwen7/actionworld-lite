# Repository map

This page explains where to find **source**, **reader-facing guides**, **immutable evidence** and **excluded generated runtime artifacts**. It does not propose moving working implementation files as part of a documentation-only change.

```text
.
├── README.md                      # Current project overview and entry point
├── app.py                         # FastAPI entrypoint
├── docs/                          # Reader guides + preserved milestone evidence
│   ├── README.md                  # Documentation navigation
│   ├── architecture.md            # Model/physics/session architecture
│   ├── models.md                  # IC-01 and M6 contracts
│   ├── evaluation.md              # TEST vs VALIDATION vs Production
│   ├── usage.md                   # Live and local use; reproduction boundary
│   ├── api.md                     # Web API contract
│   ├── deployment.md              # Vercel Container implementation
│   ├── limitations.md             # Scope, negative results, open gates
│   ├── project-structure.md       # This map
│   ├── M4_ACCEPTANCE_RECORD.md    # Original one-time TEST acceptance (preserved)
│   ├── M7_LOCAL_RELEASE.md        # Original local release record (preserved)
│   ├── M8_WEB_HANDOFF.md          # Original pre-deploy handoff (preserved)
│   └── M8_STEP3_DEPLOYMENT.md     # Original Production evidence (preserved)
├── research/                      # Research index and original M3/M6 reports
├── src/
│   ├── simulation/                # MuJoCo PushT scene and environment
│   ├── data/                      # Data generation/split/baseline scripts
│   ├── models/                    # Frozen IC-01 and component source
│   ├── train/                     # Historical training code
│   ├── research/                  # M6 residual algorithm
│   ├── eval/                      # Original M4 evaluator; DO NOT rerun TEST
│   └── demo/                      # Original local GUI demos
├── m8_web/                       # API, signed token, inference adapter, request-local physics
├── public/                       # HTML/CSS/JS only, no weights
├── deploy/m8_vercel/             # Explicit staging, Container image, remote tests
├── tests/                        # M8 synthetic/mock/real acceptance tests
├── requirements.txt              # M8 web direct dependencies
└── requirements-m3.txt           # M3-specific pins (historical scope)
```

Generated datasets and trained checkpoint artifacts under `data/` are intentionally **Git-ignored**; the public repo does not contain all runtime inputs. Some evidence files and local logs were archived outside Git as recorded in the historical milestone documents. Never add private `data/processed` model artifacts to version control as part of a documentation cleanup.

## Documentation ownership

- **README and topic guides:** living orientation documents. Update them when a later verified milestone changes the current state.
- **M3/M4/M6/M7/M8 acceptance records:** dated historical evidence. Preserve the original text, hashes, statistical scope and caveats; add clarifications via separate guides or dated addenda.
- **Source comments and tests:** implementation truth. If reader guides disagree with source or accepted evidence, flag it rather than silently modifying frozen code or metrics.
- **Personal portfolio:** may link to the live demo and this README, but should use the same scoped technical claims, not stronger unsourced ones.
