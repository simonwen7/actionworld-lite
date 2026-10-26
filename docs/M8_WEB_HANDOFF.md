# ActionWorld Lite — M8 Web implementation handoff (Step 2)

**Scope:** One Vercel FastAPI function serves real PushT + IC-01 + M6 interactions, and hosts an HTML/CSS/JS frontend. No videos, prerecorded response tables, alternate model or new research experiment. **Vercel deployment is not yet verified.**

### New files and entrypoint

- `app.py` — supported FastAPI entrypoint (auto-detection at Vercel project root)
- `m8_web/api.py` — public HTML and JSON API, request bounds and failure behavior
- `m8_web/session.py` — signed, 15-minute, 64-action, browser-carried physics snapshot + two real observed RGB frames
- `m8_web/engine.py` — request-local MuJoCo env and frame rendering; caches only *immutable* IC-01 checkpoint and fitted M6 coefficients
- `m8_web/model_adapter.py` — immutable hashes, load model, RGB conversion, one-step prediction and post-step scoring
- `public/index.html`, `public/m8-style.css`, `public/m8-app.js` — no-build static frontend
- `requirements.txt`, `.python-version` — Vercel Python 3.12 / pinned dependencies
- `tests/test_m8_synthetic.py`, `tests/test_m8_api_synthetic.py` — offline synthetic protocol and mocked HTTP tests

**No existing tracked M1–M7 files should be modified. No new package installation is performed by copying the source.**

### Local checks in Step 2

From repo root, after copying the new files:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.test_m8_synthetic -v
# Only when FastAPI AND httpx are already installed in this environment:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest tests.test_m8_api_synthetic -v
```

These tests use `numpy`, Python stdlib and synthetic arrays: they do not load real weights, datasets or any historical TEST. Source AST compilation may be done separately without executing modules. If FastAPI is installed locally, import `app` and exercise a *synthetic mocked backend*; real CPU smoke is optional in Step 2 and must **never** score real TRAIN/VAL/TEST episodes.

The **live** Vercel build, MuJoCo headless rendering, 2 GB memory limit, multi-step state restore parity, and browser E2E evidence belong to M8 Step 3.

### Owner-required actions for Step 3

1. Before public deployment, approve Git commit/push of *new M8 source only*, not original frozen M4–M7 files.
2. Provide the accepted three *runtime* artifacts to the Vercel build by an explicit owner-approved distribution mechanism: `m4_visual_v03/best_model.pt`, `m4_yaw_v04/best_model.pt`, `m6_residual_v01/model.json`, with exact SHA-256. These files are currently gitignored, so a Git-connected Vercel build **cannot run without them**.
3. Set a long random `SESSION_SECRET` (at least 32 bytes) **only** in the Vercel project's environment configuration, not in Git or a browser bundle. Confirm Hobby and `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in project settings. Never disclose secret in stdout or screenshots.
4. Confirm the necessary Linux rendering libraries exist for headless MuJoCo. If not, **stop** rather than fake predictions or install a paid alternate server.

### Correctness, security and claims

- A token signs integration-state float64 values plus two genuine RGB observations. HMAC prevents tampering but **does not prevent replay**. Two simultaneous requests using the same still-valid token can produce independent branches. The browser serializes clicks; no cross-tab exactly-once guarantees exist without server-side shared storage.
- Token expiry is fixed at initial reset (15 min). Session limit is 64 actions. Users may Reset to begin a new session.
- Model inputs never include `observe()` privileged values or the future frame. `observe()` appears only after real physics stepping for scoring and frontend visualization.
- The server fails closed when frozen inputs are absent or hashes differ. It never loads `src/demo/m5_interactive.py`, `src/demo/m7_release.py`, dataset loaders, or the one-time M4 TEST evaluator.
- Vercel request and memory compatibility and Linux `MUJOCO_GL=osmesa` are UNVERIFIED. Browser rendering is actual MuJoCo PNG plus an independent XY diagram, not a learned next RGB frame.
- M6's 13.18% validation mean-error reduction is historical VALIDATION evidence, **not** a per-action guarantee, independent TEST result, or transferable-domain claim.
- The single-function architecture intentionally avoids a second cloud service, but Hobby quotas and lack of durable rate limiting mean a heavily used public URL can exhaust free allocation. Never claim guaranteed free uptime.

### Useful references

- <https://vercel.com/docs/functions/runtimes/python>
- <https://vercel.com/docs/frameworks/backend/fastapi>
- <https://vercel.com/docs/functions/limitations>
