# Deployment — Vercel Container (M8)

**Live:** <https://actionworld-lite.vercel.app>  
**Actual runtime:** Vercel **Container**, Python 3.12 CPU-only, FastAPI, MuJoCo + OSMesa.  
**Recorded status:** Production deployed and functionally accepted on 2026-10-26; M8 final acceptance remains **CONDITIONAL** pending recorded checks.

This page is a reader/operator guide, **not an authorization to redeploy**. The authoritative dated evidence is [M8 Step 3 deployment record](M8_STEP3_DEPLOYMENT.md).

## Why a Container?

The Vercel native Python runtime could not provide the required `libOSMesa` system dependency for headless MuJoCo rendering. The project therefore uses a Debian-based Container that installs `libosmesa6`, pins Python and model dependencies, then serves the unchanged FastAPI app with Uvicorn. This was an explicit implementation change from the original Step 1 plan, not evidence that a native Python deployment succeeded.

Recorded versions: **Python 3.12.14**, **PyTorch 2.14.1+cpu**, **MuJoCo 3.17.0**, **NumPy 2.5.4**, **FastAPI 0.117.1**. Frozen artifact hashes are checked during image build and verified again before the first model inference in each runtime instance. The recorded container image was **341.8 MB compressed**.

## Allowlisted staging instead of Git-connected weights

`deploy/m8_vercel/stage.py` assembles the required source files, scene, dependency metadata, and the **three** frozen runtime artifacts into a temporary directory **outside the repository**. The two checkpoints and fitted residual JSON remain under `data/processed/`, Git-ignored, and are never placed in `public/`. Neither datasets, sealed TEST archives, other model candidates, Git metadata nor credentials belong in that staging set.

The deploy account and `SESSION_SECRET` are owner-managed. Secrets must not be echoed in console logs, copied into source, stored under public paths, or committed. Linking the Vercel project may create a local `.env.local` containing an OIDC token; inspect and remove that transient file from staging before deployment, and keep `.env*` excluded in container/deployment ignore rules.

The original record contains deployment commands and acceptance tooling. **Do not paste deployment commands blindly:** the owner must inspect the current project, staging manifest, plan and secret configuration first. All Git writes are owner-only.

## Runtime characteristics

- One FastAPI service hosts the site and API; no database or external inference service.
- Signed browser-carried sessions hold 51-element MuJoCo integration state plus two real RGB inputs; request-specific `MjData` is not shared.
- Vercel Hobby usage is bounded by included quotas. A public URL does **not** guarantee uptime or unlimited requests.
- No durable public rate limiter is implemented; token integrity does not prevent signed-token replay.
- The recorded local Linux/arm64 container peaked at **374 MiB** within a 2 GiB envelope. **Direct Vercel Production memory metrics were unavailable**, so this measurement is not falsely described as Vercel peak RSS.
- Cross-platform contact physics and rendered pixels differ slightly; see [limitations](limitations.md).

## Production verification

The dated record reports **479/479** general acceptance assertions and **416/416** contact-run assertions, plus browser checks. No model or Python runtime source changes were required to make the deployment work; the deployment scripts and documentation were new M8 files.

The initial deploy went directly to Production, skipping Preview. Restore-versus-continuous parity on **Vercel x86_64** and direct Vercel peak-memory readings remained unverified. These are not retrospectively marked PASS. See [evaluation](evaluation.md) and [full deployment record](M8_STEP3_DEPLOYMENT.md).
