# ActionWorld Lite — M8 Step 3 Vercel deployment record

**Production URL:** <https://actionworld-lite.vercel.app>
**Vercel project:** `actionworld-lite` (team `simonwen7s-projects`, **Hobby**), project ID `prj_MUa5Xg2OcM1fMrtE3hW5kinL0xc9`
**Deployment:** `dpl_CaTqyLrqYGoHEiJjAwbfyqrc7uU9`, region `iad1`, image `vcr.vercel.com/simonwen7s-projects/actionworld-lite/dockerfile@sha256:e080d4857b7559dfab6cb2b74df240b9a519a17105b1ce05f581d478220f31f0` (amd64, 341.8 MB compressed)
**Recorded:** 2026-10-26. Deployed with Vercel CLI 65.2.0 from a local staging directory, not from Git.

## Runtime

The same FastAPI app (`app.py` → `m8_web.api`) runs as one Vercel Function using the **Container runtime (beta, all plans)** on Fluid compute. The native Vercel Python runtime does not provide `libOSMesa`, which headless MuJoCo rendering needs, so `deploy/m8_vercel/Dockerfile` installs Debian `libosmesa6` and sets `MUJOCO_GL=osmesa`.

| Component | Version |
| --- | --- |
| Base image | `python:3.12.14-slim-bookworm@sha256:392307d2…` |
| Python | 3.12.14 |
| PyTorch | 2.14.1+cpu (CPU-only build of 2.14.1 from `download.pytorch.org/whl/cpu`) |
| MuJoCo | 3.17.0 (PyOpenGL 3.1.10, OSMesa) |
| NumPy | 2.5.4 |
| FastAPI / Uvicorn | 0.117.1 / 0.54.0 |

The build prints `M8 PINNED RUNTIME …` and `M8 FROZEN ARTIFACTS VERIFIED 377e56f4…` and fails if any version or frozen SHA-256 differs. `SESSION_SECRET` is a sensitive Vercel environment variable (separate values for Preview and Production); it is never in Git, the image or this file.

## Reproducing the deployment

```bash
.venv/bin/python -B deploy/m8_vercel/stage.py ~/Desktop/ActionWorldLite_M8_Step3_Deploy/stage
cd ~/Desktop/ActionWorldLite_M8_Step3_Deploy/stage
vercel link --yes --project actionworld-lite   # then delete any .env.local it writes
vercel deploy            # Preview
vercel deploy --prod     # Production
```

`stage.py` copies an explicit allowlist (the M8 web app, the frozen IC-01/M6/PushT sources it imports, and exactly the three runtime artifacts `m4_visual_v03/best_model.pt`, `m4_yaw_v04/best_model.pt`, `m6_residual_v01/model.json`) into a directory outside the repository, checks every frozen SHA-256, and refuses anything else. No dataset, TEST archive, `validation_report.json`, Git metadata or virtual environment is staged. The artifacts are never under `public/`, and the app serves only `/`, `/m8-style.css`, `/m8-app.js`, `/api/health`, `/api/session` and `/api/action`.

Black-box acceptance against any running deployment:

```bash
.venv/bin/python -B deploy/m8_vercel/remote_acceptance.py https://actionworld-lite.vercel.app EVIDENCE.json
```

It recomputes every served action locally with the same frozen IC-01, fitted M6 and MuJoCo scene, starting from the physics state and frames carried in the server's own token, and needs no signing secret.

## Production acceptance results (2026-10-26)

- `remote_acceptance.py`: **479/479 checks passed**, 138 HTTP calls. Reset returns scored count 0, physics time 0.05 s, a 256×256 MuJoCo PNG with the frozen IC-01 orange and weak-blue masks present. Each action returns exactly one IC-01 forward and one physics step, time advances by 0.05 s, reported cm errors match a recomputation from the returned coordinates within 1e-9 cm, and the token carries the rolled RGB pair and the new 51-value integration state.
- A contact sequence (14 actions, 10 predicted contacts): **416/416 checks passed**. Corrections 2.48–4.00 mm when contact was predicted, exactly 0 otherwise.
- Rejections: tampered signature or payload → 401, 15 mm / float / unknown direction → 422, extra or missing fields → 400, 70 KB body → 413, 65th action → 409. No rejection returns a token. Model, source and artifact paths return 404.
- Replaying an accepted token is deterministic and accepted: HMAC protects integrity, **not** single use.
- Two interleaved sessions with different sequences equal their solo runs bit-for-bit.
- Browser on the production URL: Reset and real clicks over 5/10/20 mm in all four directions, all controls disabled during each request, three same-tick clicks send one request, a forced 503 leaves token, step and pose unchanged, a tampered token is reported and not committed, Reset returns to step 0 and t = 0.05 s, two tabs hold independent sessions. The page loads only same-origin resources and sets no cookies.
- Latency: action median 1.05 s (0.67–1.31 s); cold Reset 6.9–7.6 s after idle; warm Reset 1.2–1.4 s. Instances scale down after roughly 40 s idle, so cold starts are common.
- Runtime logs: info level only, no traceback, OOM or runtime-failure record.

## Measured limitations

- **Rendering differs from macOS.** Linux OSMesa frames differ from the macOS frames used for training and validation at a few edge pixels (up to 73 intensity levels, about 15 of 4,096 model pixels per frame). For identical physical states this moves IC-01 predicted XY by up to 2.97 mm and M6 corrections by up to 0.21 mm. Given identical frames, Vercel and macOS IC-01/M6 outputs agree within 6e-9 m.
- **x86_64 contact physics differs from arm64.** Without contact, Vercel and macOS states agree within 1e-13. From the first contact step, x86_64 MuJoCo diverges by about 70 µm per step in block XY (about 0.9 mm after 9 contact steps). Restore-versus-continuous parity through contact is bit-exact on macOS arm64 and Linux arm64. It was not directly measured on x86_64, because neither PyTorch nor MuJoCo x86_64 wheels run under this Mac's amd64 emulation.
- **Memory.** Vercel function memory metrics require Observability Plus, which Hobby cannot buy. The identical image peaked at 374 MiB under a 2 GiB / 1 CPU container limit locally, and Vercel reported no OOM.
- **Deployment order.** The project's first CLI deployment was assigned to Production automatically. It was then validated in place; there was no separate earlier Preview.
- Live per-action errors are not the M6 VALIDATION result. On the contact sequence above, M6 mean motion error (0.744 cm) was higher than IC-01 (0.529 cm). The 13.18% figure remains a historical VALIDATION result.
- Hobby is limited to non-commercial use and pauses when its monthly allowance is used; there is no durable rate limiting.
