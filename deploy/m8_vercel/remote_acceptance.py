"""M8 Step 3 black-box acceptance of a running deployment (container or Vercel URL).

Usage (from the repository root):
    .venv/bin/python -B deploy/m8_vercel/remote_acceptance.py BASE_URL EVIDENCE_JSON

The server is exercised only through public HTTP. Every scored response is
checked against an independent local recomputation with the same frozen IC-01,
fitted M6 and MuJoCo scene, starting from the physics state and real frames
carried inside the server's own token. The signing secret is never needed.
No dataset episode, TEST archive or M6 experiment is touched.
"""
from __future__ import annotations

import base64
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from m8_web import engine, model_adapter  # noqa: E402

OFFSETS = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
SEQUENCE_A = (("right", 10), ("up", 5), ("left", 20), ("down", 10), ("up", 20), ("left", 5),
              ("right", 5), ("down", 20), ("right", 20), ("up", 10), ("left", 10), ("down", 5))
SEQUENCE_B = (("up", 20), ("right", 5), ("down", 5), ("left", 10), ("up", 10), ("right", 20))
SEQUENCE_C = (("down", 20), ("left", 5), ("down", 10), ("right", 10), ("up", 5), ("left", 20))
HIDDEN = ("SESSION_SECRET", "state_dict", "coef", "best_model", "model.json", "/app/", "/Users/",
          "validation_report", "Traceback")
CLOSED_PATHS = (
    "/data/processed/m4_visual_v03/best_model.pt", "/data/processed/m4_yaw_v04/best_model.pt",
    "/data/processed/m6_residual_v01/model.json", "/m8_web/model_adapter.py", "/m8_web/session.py",
    "/app.py", "/Dockerfile", "/requirements.txt", "/src/simulation/pusht_scene.xml",
    "/public/index.html", "/docs", "/redoc", "/openapi.json", "/.env", "/.vercel/project.json",
)


class Http:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.calls = []

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                status, text, headers = resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            status, text, headers = exc.code, exc.read().decode("utf-8", "replace"), dict(exc.headers)
        elapsed = time.perf_counter() - start
        self.calls.append({"method": method, "path": path, "status": status, "seconds": round(elapsed, 4)})
        return status, text, headers, elapsed


def png_rgb(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n"), "not a PNG"
    pos, idat, width, height = 8, b"", None, None
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag, chunk = raw[pos + 4:pos + 8], raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, *_ = struct.unpack(">2I5B", chunk)
            assert (depth, color) == (8, 2), "unexpected PNG layout"
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    scan = zlib.decompress(idat)
    stride = width * 3 + 1
    rows = [np.frombuffer(scan[y * stride + 1:(y + 1) * stride], dtype=np.uint8) for y in range(height)]
    assert all(scan[y * stride] == 0 for y in range(height)), "unexpected PNG filter"
    return np.stack(rows).reshape(height, width, 3)


def token_payload(token):
    encoded = token.split(".")[0]
    payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    state = np.frombuffer(base64.urlsafe_b64decode(payload["physics"] + "=" * (-len(payload["physics"]) % 4)), dtype="<f8")
    frames = zlib.decompress(base64.urlsafe_b64decode(payload["rgb"] + "=" * (-len(payload["rgb"]) % 4)))
    rgb = np.frombuffer(frames, dtype=np.uint8).reshape(2, 64, 64, 3)
    return payload, state.copy(), rgb[0].copy(), rgb[1].copy()


def masks(image):
    small = image[::4, ::4].astype(np.float64) / 255.0
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    return int(((r > g + .09) & (g > b + .04) & (r > .40)).sum()), int(((b > r + .06) & (b > g + .04) & (b > .25)).sum())


def visible_text(text):
    """Response text without the opaque base64 PNG and token, which can contain any 4-letter run."""
    body = json.loads(text)
    return json.dumps({k: v for k, v in body.items() if k not in ("real_rgb_png", "session_token")})


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


class Checker:
    def __init__(self):
        self.failures = []
        self.count = 0

    def check(self, condition, message):
        self.count += 1
        if not condition:
            self.failures.append(message)


def local_step(state, previous, current, direction, mm):
    """Independent Mac recomputation: one IC-01 forward, M6, one MuJoCo step."""
    torch, mj, Env, model, fitted = engine.get_backend()
    delta = tuple(axis * mm / 1000.0 for axis in OFFSETS[direction])
    prediction = model_adapter.infer(model, fitted, previous, current, delta, torch)
    env = Env()
    try:
        engine._restore(mj, env, state)
        before = env.observe()
        env.step(delta)
        mj.mj_forward(env.model, env.data)
        after = env.observe()
        frame = model_adapter.stride64(env.render())
        return prediction, before, after, engine._state(mj, env), frame
    finally:
        env.close()


def run_session(http, ck, sequence, label, evidence, parity):
    status, text, headers, elapsed = http.request("POST", "/api/session", {})
    ck.check(status == 200, f"{label} reset status {status}: {text[:200]}")
    body = json.loads(text)
    ck.check(headers.get("Cache-Control", headers.get("cache-control")) == "no-store", f"{label} reset cache header")
    ck.check(body["scored_actions"] == 0, f"{label} reset scored {body['scored_actions']}")
    ck.check(abs(body["physics_time_s"] - 0.05) < 1e-9, f"{label} reset time {body['physics_time_s']}")
    ck.check(body["prediction"] is None and body["metrics"] is None, f"{label} reset carries a prediction")
    image = png_rgb(body["real_rgb_png"])
    orange, blue = masks(image)
    ck.check(image.shape == (256, 256, 3) and image.std() > 5, f"{label} reset image invalid")
    ck.check(6 <= orange <= 180 and 1 <= blue <= 64, f"{label} reset masks orange={orange} blue={blue}")
    payload, state, previous, current = token_payload(body["session_token"])
    ck.check(payload["v"] == 1 and payload["seq"] == 0 and payload["steps"] == 0, f"{label} token counters")
    ck.check(payload["fp"] == model_adapter.FINGERPRINT, f"{label} artifact fingerprint mismatch")
    ck.check(payload["nstate"] == 51 and state.shape == (51,), f"{label} state length {state.shape}")
    ck.check(np.array_equal(current, image[::4, ::4]), f"{label} token current frame differs from returned image")
    for hidden in HIDDEN:
        ck.check(hidden not in visible_text(text), f"{label} reset leaks {hidden}")

    _, mj, Env, _, _ = engine.get_backend()
    env = Env()
    try:
        env.reset()
        local_prev = model_adapter.stride64(env.render())
        env.step((0.0, 0.0))
        mj.mj_forward(env.model, env.data)
        local_cur = model_adapter.stride64(env.render())
        parity["reset_state_max_abs"] = max(parity.get("reset_state_max_abs", 0.0), max_abs(engine._state(mj, env), state))
        parity["reset_frame_max_abs"] = max(parity.get("reset_frame_max_abs", 0), int(np.max(np.abs(local_prev.astype(int) - previous.astype(int)))),
                                            int(np.max(np.abs(local_cur.astype(int) - current.astype(int)))))
    finally:
        env.close()
    mac_frames = [local_prev, local_cur]

    token, prior_actual, prior_png = body["session_token"], body["actual"], body["real_rgb_png"]
    steps = [{"reset_seconds": round(elapsed, 4), "orange_px64": orange, "blue_px64": blue}]
    for index, (direction, mm) in enumerate(sequence, start=1):
        payload, state, previous, current = token_payload(token)
        status, text, headers, elapsed = http.request(
            "POST", "/api/action", {"session_token": token, "step_size_mm": mm, "direction": direction})
        ck.check(status == 200, f"{label} action {index} status {status}: {text[:200]}")
        if status != 200:
            break
        out = json.loads(text)
        for hidden in HIDDEN:
            ck.check(hidden not in visible_text(text), f"{label} action {index} leaks {hidden}")
        pred, metrics = out["prediction"], out["metrics"]
        ck.check(out["scored_actions"] == index, f"{label} action {index} scored {out['scored_actions']}")
        ck.check(abs(out["physics_time_s"] - 0.05 * (index + 1)) < 1e-9, f"{label} action {index} time {out['physics_time_s']}")
        ck.check(out["ic01_forward_calls"] == 1 and out["physics_step_calls"] == 1, f"{label} action {index} call counts")
        ck.check(out["prediction_source"] == "two real RGB before action", f"{label} action {index} source")
        ck.check(out["step_size_mm"] == mm and out["direction"] == direction, f"{label} action {index} echo")
        expected_delta = [axis * mm / 1000.0 for axis in OFFSETS[direction]]
        ck.check(max_abs(out["action_m"], expected_delta) == 0.0, f"{label} action {index} action_m")
        correction = float(np.linalg.norm(pred["correction_m"]))
        ck.check(correction <= 0.004000000001, f"{label} action {index} correction {correction}")
        if not pred["predicted_contact"]:
            ck.check(correction == 0.0, f"{label} action {index} no-contact correction {correction}")
        actual = np.asarray(metrics["actual_xy_m"])
        ck.check(max_abs(actual, out["actual"]["actual_xy_m"]) == 0.0, f"{label} action {index} actual mismatch")
        motion = actual - np.asarray(prior_actual["actual_xy_m"])
        recomputed = {
            "ic01_next_xy_error_cm": float(np.linalg.norm(np.asarray(pred["ic01_xy_m"]) - actual) * 100),
            "m6_next_xy_error_cm": float(np.linalg.norm(np.asarray(pred["m6_xy_m"]) - actual) * 100),
            "ic01_motion_error_cm": float(np.linalg.norm(np.asarray(pred["ic01_delta_m"]) - motion) * 100),
            "m6_motion_error_cm": float(np.linalg.norm(np.asarray(pred["m6_delta_m"]) - motion) * 100),
        }
        for key, value in recomputed.items():
            ck.check(abs(metrics[key] - value) <= 1e-9, f"{label} action {index} {key} {metrics[key]} vs {value}")
        ck.check(metrics["contact_correct"] == (pred["predicted_contact"] == metrics["actual_contact"]), f"{label} action {index} contact_correct")
        ck.check(max_abs(np.asarray(pred["m6_xy_m"]) - np.asarray(pred["ic01_xy_m"]), pred["correction_m"]) <= 1e-12,
                 f"{label} action {index} M6 = IC-01 + correction")
        image = png_rgb(out["real_rgb_png"])
        ck.check(out["real_rgb_png"] != prior_png, f"{label} action {index} image not refreshed")
        new_payload, new_state, new_prev, new_cur = token_payload(out["session_token"])
        ck.check(new_payload["seq"] == index and new_payload["sid"] == payload["sid"], f"{label} action {index} token chain")
        ck.check(np.array_equal(new_prev, current), f"{label} action {index} previous frame not rolled forward")
        ck.check(np.array_equal(new_cur, image[::4, ::4]), f"{label} action {index} next frame differs from returned image")

        local_pred, local_before, local_after, local_state, local_frame = local_step(state, previous, current, direction, mm)
        parity["ic01_xy_max_abs_m"] = max(parity.get("ic01_xy_max_abs_m", 0.0), max_abs(local_pred["ic01_xy_m"], pred["ic01_xy_m"]))
        parity["m6_xy_max_abs_m"] = max(parity.get("m6_xy_max_abs_m", 0.0), max_abs(local_pred["m6_xy_m"], pred["m6_xy_m"]))
        parity["correction_max_abs_m"] = max(parity.get("correction_max_abs_m", 0.0), max_abs(local_pred["correction_m"], pred["correction_m"]))
        parity["contact_pred_mismatches"] = parity.get("contact_pred_mismatches", 0) + int(local_pred["predicted_contact"] != pred["predicted_contact"])
        parity["physics_state_max_abs"] = max(parity.get("physics_state_max_abs", 0.0), max_abs(local_state, new_state))
        parity["actual_xy_max_abs_m"] = max(parity.get("actual_xy_max_abs_m", 0.0), max_abs(local_after["block_xy"], actual))
        parity["actual_contact_mismatches"] = parity.get("actual_contact_mismatches", 0) + int(bool(local_after["pusher_contact"]) != metrics["actual_contact"])
        parity["frame_max_abs"] = max(parity.get("frame_max_abs", 0), int(np.max(np.abs(local_frame.astype(int) - new_cur.astype(int)))))
        parity["frame_pixels_differing"] = parity.get("frame_pixels_differing", 0) + int(np.any(local_frame != new_cur, axis=2).sum())
        parity["actions_compared"] = parity.get("actions_compared", 0) + 1

        # Same physics state, but frames rendered by this Mac's MuJoCo (as in M5/M7 and training).
        torch, _, _, model, fitted = engine.get_backend()
        delta = tuple(axis * mm / 1000.0 for axis in OFFSETS[direction])
        mac_pred = model_adapter.infer(model, fitted, mac_frames[0], mac_frames[1], delta, torch)
        render_effect = parity.setdefault("mac_rendered_inputs_vs_served", {
            "ic01_xy_max_abs_m": 0.0, "m6_xy_max_abs_m": 0.0, "correction_max_abs_m": 0.0, "contact_pred_mismatches": 0})
        render_effect["ic01_xy_max_abs_m"] = max(render_effect["ic01_xy_max_abs_m"], max_abs(mac_pred["ic01_xy_m"], pred["ic01_xy_m"]))
        render_effect["m6_xy_max_abs_m"] = max(render_effect["m6_xy_max_abs_m"], max_abs(mac_pred["m6_xy_m"], pred["m6_xy_m"]))
        render_effect["correction_max_abs_m"] = max(render_effect["correction_max_abs_m"], max_abs(mac_pred["correction_m"], pred["correction_m"]))
        render_effect["contact_pred_mismatches"] += int(mac_pred["predicted_contact"] != pred["predicted_contact"])
        mac_frames = [mac_frames[1], local_frame]

        steps.append({
            "index": index, "direction": direction, "mm": mm, "seconds": round(elapsed, 4),
            "physics_time_s": out["physics_time_s"], "predicted_contact": pred["predicted_contact"],
            "actual_contact": metrics["actual_contact"], "correction_mm": round(correction * 1000, 6),
            "ic01_next_xy_error_cm": metrics["ic01_next_xy_error_cm"], "m6_next_xy_error_cm": metrics["m6_next_xy_error_cm"],
            "ic01_motion_error_cm": metrics["ic01_motion_error_cm"], "m6_motion_error_cm": metrics["m6_motion_error_cm"],
            "actual_xy_m": metrics["actual_xy_m"], "pusher_xy_m": metrics["pusher_xy_m"],
        })
        token, prior_actual, prior_png = out["session_token"], out["actual"], out["real_rgb_png"]
    evidence[label] = steps
    return body["session_token"], token


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    base, out_path = sys.argv[1], Path(sys.argv[2]).expanduser()
    if out_path.exists():
        raise SystemExit(f"Evidence file already exists: {out_path}")
    http, ck = Http(base), Checker()
    evidence = {"base_url": base, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "expected_fingerprint": model_adapter.FINGERPRINT}

    status, text, headers, elapsed = http.request("GET", "/")
    evidence["cold_first_get_seconds"] = round(elapsed, 4)
    ck.check(status == 200 and "REAL SIMULATION" in text and "13.18%" in text and "NOT a new held-out TEST" in text,
             f"home page status {status}")
    for path in ("/m8-app.js", "/m8-style.css"):
        status, *_ = http.request("GET", path)
        ck.check(status == 200, f"{path} status {status}")
    status, text, *_ = http.request("GET", "/api/health")
    ck.check(status == 200 and json.loads(text) == {"app": "ActionWorld Lite M8", "mode": "real-inference-only", "ready": True},
             f"health {status} {text[:200]}")
    evidence["closed_paths"] = {}
    for path in CLOSED_PATHS:
        status, text, *_ = http.request("GET", path)
        evidence["closed_paths"][path] = status
        ck.check(status in (404, 405) and len(text) < 200, f"closed path {path} returned {status}")

    parity = {}
    first_token_a, last_token_a = run_session(http, ck, SEQUENCE_A, "session_A", evidence, parity)

    # Interleave a second session with a different sequence; compare to its local continuous chain.
    status, text, *_ = http.request("POST", "/api/session", {})
    token_b = json.loads(text)["session_token"]
    status, text, *_ = http.request("POST", "/api/session", {})
    token_c = json.loads(text)["session_token"]
    b_results, c_results = [], []
    seq_c = SEQUENCE_C
    for (dir_b, mm_b), (dir_c, mm_c) in zip(SEQUENCE_B, seq_c):
        for label, direction, mm, results in (("B", dir_b, mm_b, b_results), ("C", dir_c, mm_c, c_results)):
            current_token = token_b if label == "B" else token_c
            status, text, *_ = http.request("POST", "/api/action",
                                            {"session_token": current_token, "step_size_mm": mm, "direction": direction})
            ck.check(status == 200, f"interleaved {label} status {status}")
            out = json.loads(text)
            results.append(out)
            if label == "B":
                token_b = out["session_token"]
            else:
                token_c = out["session_token"]
    _, mj, Env, _, _ = engine.get_backend()
    isolation = {}
    for label, sequence, results in (("B", SEQUENCE_B, b_results), ("C", seq_c, c_results)):
        status, text, *_ = http.request("POST", "/api/session", {})
        solo = json.loads(text)["session_token"]
        identical = True
        for (direction, mm), interleaved in zip(sequence, results):
            status, text, *_ = http.request("POST", "/api/action",
                                            {"session_token": solo, "step_size_mm": mm, "direction": direction})
            alone = json.loads(text)
            solo = alone["session_token"]
            _, state_i, prev_i, cur_i = token_payload(interleaved["session_token"])
            _, state_s, prev_s, cur_s = token_payload(solo)
            identical &= (np.array_equal(state_i, state_s) and np.array_equal(prev_i, prev_s)
                          and np.array_equal(cur_i, cur_s) and alone["prediction"] == interleaved["prediction"]
                          and alone["metrics"] == interleaved["metrics"])
        env = Env()
        try:
            env.reset(); env.step((0.0, 0.0)); mj.mj_forward(env.model, env.data)
            worst = 0.0
            for (direction, mm), out in zip(sequence, results):
                env.step(tuple(axis * mm / 1000.0 for axis in OFFSETS[direction]))
                mj.mj_forward(env.model, env.data)
                worst = max(worst, max_abs(env.observe()["block_xy"], out["actual"]["actual_xy_m"]),
                            max_abs(env.observe()["pusher_xy"], out["actual"]["pusher_xy_m"]))
        finally:
            env.close()
        isolation[label] = {"actions": len(results), "interleaved_equals_solo_bitwise": bool(identical),
                            "local_mac_continuous_vs_served_max_abs_m": worst,
                            "final_pusher_xy_m": results[-1]["actual"]["pusher_xy_m"]}
        ck.check(identical, f"session {label} interleaved results differ from a solo run")
    ck.check(isolation["B"]["final_pusher_xy_m"] != isolation["C"]["final_pusher_xy_m"], "interleaved sessions share state")
    evidence["interleaved_isolation"] = isolation

    # Replaying an accepted token is deterministic and NOT rejected: HMAC is not single-use.
    replay_body = {"session_token": first_token_a, "step_size_mm": 10, "direction": "right"}
    s1, t1, *_ = http.request("POST", "/api/action", replay_body)
    s2, t2, *_ = http.request("POST", "/api/action", replay_body)
    same = s1 == s2 == 200 and json.loads(t1)["session_token"] == json.loads(t2)["session_token"]
    ck.check(same, "replay of the same token is not deterministic")
    evidence["replay_same_token_identical_and_accepted"] = same

    head, signature = last_token_a.split(".")
    tampered = head + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
    payload_char = ("A" if head[10] != "A" else "B")
    tampered_payload = head[:10] + payload_char + head[11:] + "." + signature
    rejections = {}
    for name, body in (
        ("signature_char", {"session_token": tampered, "step_size_mm": 10, "direction": "right"}),
        ("payload_char", {"session_token": tampered_payload, "step_size_mm": 10, "direction": "right"}),
        ("step_15mm", {"session_token": last_token_a, "step_size_mm": 15, "direction": "right"}),
        ("step_float", {"session_token": last_token_a, "step_size_mm": 10.0, "direction": "up"}),
        ("direction_zero", {"session_token": last_token_a, "step_size_mm": 10, "direction": "zero"}),
        ("direction_case", {"session_token": last_token_a, "step_size_mm": 10, "direction": "Up"}),
        ("extra_field", {"session_token": last_token_a, "step_size_mm": 10, "direction": "up", "x": 1}),
        ("missing_token", {"step_size_mm": 10, "direction": "up"}),
    ):
        status, text, *_ = http.request("POST", "/api/action", body)
        rejections[name] = status
        ck.check(status in (400, 401, 422) and "session_token" not in json.loads(text), f"{name} not rejected: {status}")
    status, text, *_ = http.request("POST", "/api/action", raw=b'{"session_token":"' + b"a" * 70000 + b'"}')
    rejections["body_70000_bytes"] = status
    ck.check(status == 413, f"oversized body status {status}")
    status, text, *_ = http.request("POST", "/api/action", raw=b"not json")
    rejections["malformed_json"] = status
    ck.check(status == 400, f"malformed json status {status}")
    evidence["rejections"] = rejections

    status, text, *_ = http.request("POST", "/api/session", {})
    token = json.loads(text)["session_token"]
    limit_status = None
    for index in range(1, 66):
        status, text, *_ = http.request("POST", "/api/action",
                                        {"session_token": token, "step_size_mm": 5, "direction": "up" if index % 2 else "down"})
        if index <= 64:
            ck.check(status == 200, f"limit run action {index} status {status}")
            token = json.loads(text)["session_token"]
            if index == 64:
                ck.check(json.loads(text)["scored_actions"] == 64, "limit run count")
        else:
            limit_status = status
            ck.check(status == 409 and "session_token" not in json.loads(text), f"65th action status {status}")
    evidence["action_65_status"] = limit_status

    evidence["parity_against_local_mac_recomputation"] = parity
    evidence["http_calls"] = len(http.calls)
    evidence["action_seconds"] = sorted(c["seconds"] for c in http.calls if c["path"] == "/api/action" and c["status"] == 200)
    evidence["checks"] = ck.count
    evidence["failures"] = ck.failures
    evidence["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in (
        "base_url", "checks", "failures", "http_calls", "cold_first_get_seconds", "closed_paths",
        "rejections", "action_65_status", "replay_same_token_identical_and_accepted",
        "interleaved_isolation", "parity_against_local_mac_recomputation")}, indent=2, sort_keys=True))
    times = evidence["action_seconds"]
    print(f"action latency s: min {times[0]:.3f} median {times[len(times) // 2]:.3f} max {times[-1]:.3f} n={len(times)}")
    print("M8 REMOTE ACCEPTANCE:", "PASS" if not ck.failures else f"FAIL ({len(ck.failures)})")
    sys.exit(0 if not ck.failures else 1)


if __name__ == "__main__":
    main()
