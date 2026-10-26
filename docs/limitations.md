# Scope, limitations and open checks

## Scientific scope

- **One-step state prediction only.** IC-01 predicts next object position, yaw, and contact for one commanded action; it does not synthesize future RGB images or demonstrate a validated autonomous long-horizon learned rollout.
- **Fixed synthetic domain.** Simplified MuJoCo PushT-style geometry, colors, top-camera placement, action range and physical parameters. No real-robot, unseen-object, camera-transfer, or physical-domain generalization claim.
- **Hybrid model.** IC-01 combines trained neural components, fixed visual calibration, contact geometry and gating; it is not a fully end-to-end learned general-purpose simulator.
- **Evaluation scope.** M4 TEST is held out by episode, but drawn from the same simulator family. M6's 13.18% headline is one-pass **VALIDATION**, not a new independent TEST. M6 gets worse in documented static/no-contact cohorts and in the recorded live contact sequence.
- **Training/data reproducibility.** Generated datasets, `.pt` checkpoints and fitted residual `model.json` are not in the public repo. There is no public fresh-clone end-to-end model reproduction claim or settled checkpoint redistribution/license policy.

## Measured platform differences

- **Rendered input differences.** macOS and Linux OSMesa render small differences near edges. With the same physical state, they can shift IC-01 XY prediction by up to **2.97 mm** in the recorded comparison.
- **Contact physics differences.** Linux/x86_64 differs from macOS/arm64 during contact by about **70 μm per step** initially and roughly **0.9 mm after nine contact steps** in the recorded run. This does **not** itself establish a restore-token bug: cross-architecture divergence and same-runtime restore parity are different questions.
- **Latency/availability.** Recorded Production median per-action latency was **1.05 s**. Cold reset could take **6.9–7.6 s**; Hobby resource quotas and scale-down may affect future operation.

## Outstanding M8 acceptance checks

At the time of the 2026-10-26 production record, final status was **CONDITIONAL** for three separately documented reasons:

1. **Vercel x86_64 restore-versus-continuous physics parity** was not directly measured through the contact sequence. arm64 parity and deterministic token replay are not substitutes for that exact test.
2. **Preview was skipped before the original Production deployment.** A later Preview can validate the workflow going forward but cannot change the historical order.
3. **Direct Vercel-side peak-memory measurement** was unavailable on Hobby. The local equivalent image showed 374 MiB peak memory under a 2 GiB limit; logs showed no OOM, but that is indirect production evidence.

Only a new dated, evidence-backed verification record may close these items. Do not silently upgrade the status to PASS.

## Security and operating boundaries

The token HMAC detects tampering; it is not encrypted, revocable single-use storage, or protection against all replay/race conditions. The browser locks buttons, but cross-tab or external clients can reuse a still-valid token. There is no durable rate limit or uptime service-level agreement. Never publish signing keys, private runtime artifacts, sealed TEST archives or internal deployment credentials.

Historical status language is preserved in the original acceptance/handoff records. For current state, use the [main README](../README.md), [documentation index](README.md) and any subsequent dated closure record.
