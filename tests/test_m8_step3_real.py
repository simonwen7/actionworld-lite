"""M8 Step 3 local checks: real HTTP contract and restore-versus-continuous parity.

These tests load the frozen CPU weights and MuJoCo. They do not open dataset
episodes, the sealed TEST evaluator, or M6 training.
"""
from __future__ import annotations

import base64
import os
import struct
import unittest
import zlib
from unittest import mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SECRET = "m8-step3-local-http-" + "k" * 48
OFFSETS = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
SEQUENCE = (
    ("right", 10),
    ("up", 5),
    ("left", 20),
    ("down", 10),
    ("up", 20),
    ("left", 5),
    ("right", 5),
    ("down", 20),
)


def _png_rgb(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("Response image is not a PNG")
    pos = 8
    width = height = None
    idat = b""
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        chunk = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, inter = struct.unpack(">2I5B", chunk)
            if (depth, color, comp, filt, inter) != (8, 2, 0, 0, 0):
                raise AssertionError("Unexpected PNG layout")
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    scan = zlib.decompress(idat)
    rows = []
    stride = width * 3
    cursor = 0
    for _ in range(height):
        if scan[cursor] != 0:
            raise AssertionError("PNG scanline filter is not the encoder's filter 0")
        cursor += 1
        rows.append(np.frombuffer(scan[cursor:cursor + stride], dtype=np.uint8).copy())
        cursor += stride
    image = np.stack(rows).reshape(height, width, 3)
    return image


def _integration(mj, env):
    spec = mj.mjtState.mjSTATE_INTEGRATION
    out = np.empty(mj.mj_stateSize(env.model, spec), dtype=np.float64)
    mj.mj_getState(env.model, env.data, out, spec)
    return out


class RealHttpAndParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SESSION_SECRET"] = SECRET
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        from fastapi.testclient import TestClient
        from m8_web.api import app
        cls.client = TestClient(app)
        cls.engine = __import__("m8_web.engine", fromlist=["engine"])
        cls.session = __import__("m8_web.session", fromlist=["session"])
        cls.adapter = __import__("m8_web.model_adapter", fromlist=["model_adapter"])

    def test_page_contract_and_static_model_routes_closed(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        text = page.text
        self.assertIn("REAL SIMULATION", text)
        self.assertIn("XY WORLD COORDINATES", text)
        self.assertIn("13.18%", text)
        self.assertIn("VALIDATION", text)
        self.assertIn("NOT a new held-out TEST", text)
        self.assertIn("0.156625 cm", text)
        self.assertIn("0.135983 cm", text)
        script = (ROOT / "public/m8-app.js").read_text(encoding="utf-8")
        self.assertIn("if(busy||!token)return", script)
        self.assertIn("const original=token", script)
        self.assertIn("Token unchanged", script)
        self.assertNotIn("prerecorded", script.lower().replace("no prerecorded", ""))
        for path in (
            "/data/processed/m4_visual_v03/best_model.pt",
            "/data/processed/m4_yaw_v04/best_model.pt",
            "/data/processed/m6_residual_v01/model.json",
            "/m8_web/model_adapter.py",
        ):
            denied = self.client.get(path)
            self.assertEqual(denied.status_code, 404, path)
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"app": "ActionWorld Lite M8", "mode": "real-inference-only", "ready": True})
        self.assertNotIn(SECRET, health.text)
        self.assertNotIn("best_model", health.text)

    def test_real_session_and_one_scored_action(self):
        start = self.client.post("/api/session", json={})
        self.assertEqual(start.status_code, 200, start.text)
        body = start.json()
        self.assertEqual(body["scored_actions"], 0)
        self.assertAlmostEqual(body["physics_time_s"], 0.05, delta=1e-9)
        self.assertIsNone(body["prediction"])
        self.assertTrue(body["real_rgb_png"].startswith("data:image/png;base64,"))
        image = _png_rgb(body["real_rgb_png"])
        self.assertEqual(image.shape, (256, 256, 3))
        self.assertGreater(float(image.std()), 5.0)
        # The frozen IC-01 orange / weak-blue segmentation rules on the 64 px model view.
        small = image[::4, ::4].astype(np.float64) / 255.0
        r, g, b = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        orange = int(((r > g + 0.09) & (g > b + 0.04) & (r > 0.40)).sum())
        blue = int(((b > r + 0.06) & (b > g + 0.04) & (b > 0.25)).sum())
        self.assertTrue(6 <= orange <= 180 and 1 <= blue <= 64,
                        f"Reset image lacks the orange block or blue pusher ({orange}, {blue})")
        self.assertNotIn(SECRET, start.text)
        self.assertNotIn("best_model", start.text)
        self.assertNotIn("correction_cap", start.text)
        token = body["session_token"]
        verified = self.session.verify(token, SECRET)
        self.assertEqual(verified["seq"], 0)
        self.assertEqual(verified["fp"], self.adapter.FINGERPRINT)

        action = self.client.post("/api/action", json={
            "session_token": token, "step_size_mm": 10, "direction": "right",
        })
        self.assertEqual(action.status_code, 200, action.text)
        step = action.json()
        self.assertEqual(step["scored_actions"], 1)
        self.assertEqual(step["ic01_forward_calls"], 1)
        self.assertEqual(step["physics_step_calls"], 1)
        self.assertEqual(step["prediction_source"], "two real RGB before action")
        self.assertAlmostEqual(step["physics_time_s"], 0.10, delta=1e-9)
        self.assertNotEqual(step["real_rgb_png"], body["real_rgb_png"])
        self.assertNotEqual(step["session_token"], token)
        pred = step["prediction"]
        metrics = step["metrics"]
        correction = np.asarray(pred["correction_m"], dtype=np.float64)
        self.assertLessEqual(float(np.linalg.norm(correction)), 0.004000000001)
        if not pred["predicted_contact"]:
            self.assertEqual(float(np.linalg.norm(correction)), 0.0)
        actual = np.asarray(metrics["actual_xy_m"], dtype=np.float64)
        before = np.asarray(body["actual"]["actual_xy_m"], dtype=np.float64)
        delta = actual - before
        ic01_xy = float(np.linalg.norm(np.asarray(pred["ic01_xy_m"]) - actual) * 100)
        m6_xy = float(np.linalg.norm(np.asarray(pred["m6_xy_m"]) - actual) * 100)
        ic01_motion = float(np.linalg.norm(np.asarray(pred["ic01_delta_m"]) - delta) * 100)
        m6_motion = float(np.linalg.norm(np.asarray(pred["m6_delta_m"]) - delta) * 100)
        self.assertAlmostEqual(metrics["ic01_next_xy_error_cm"], ic01_xy, delta=1e-6)
        self.assertAlmostEqual(metrics["m6_next_xy_error_cm"], m6_xy, delta=1e-6)
        self.assertAlmostEqual(metrics["ic01_motion_error_cm"], ic01_motion, delta=1e-6)
        self.assertAlmostEqual(metrics["m6_motion_error_cm"], m6_motion, delta=1e-6)
        self.assertNotIn(SECRET, action.text)
        for hidden in ("coef", "SESSION_SECRET", "state_dict", "/Users/"):
            self.assertNotIn(hidden, action.text)

    def test_rejected_tokens_and_inputs_do_not_mint_state(self):
        start = self.client.post("/api/session", json={}).json()
        token = start["session_token"]
        head, signature = token.split(".")
        flipped = ("A" if signature[:1] != "A" else "B") + signature[1:]
        tampered = self.client.post("/api/action", json={
            "session_token": head + "." + flipped, "step_size_mm": 10, "direction": "right",
        })
        self.assertEqual(tampered.status_code, 401)
        self.assertNotIn("session_token", tampered.json())

        output, state, previous, current = self.engine.reset_scene()
        expired_payload = self.session.new_payload(
            state, previous, current, self.adapter.FINGERPRINT, now=1,
        )
        expired = self.client.post("/api/action", json={
            "session_token": self.session.sign(expired_payload, SECRET),
            "step_size_mm": 10, "direction": "right",
        })
        self.assertEqual(expired.status_code, 401, expired.text)
        self.assertNotIn("session_token", expired.json())

        payload = self.session.new_payload(state, previous, current, self.adapter.FINGERPRINT)
        for _ in range(self.session.MAX_STEPS):
            payload = self.session.next_payload(payload, state, previous, current)
        self.assertEqual(payload["steps"], 64)
        limited = self.client.post("/api/action", json={
            "session_token": self.session.sign(payload, SECRET),
            "step_size_mm": 5, "direction": "up",
        })
        self.assertEqual(limited.status_code, 409, limited.text)
        self.assertNotIn("session_token", limited.json())

        for bad in (
            {"session_token": token, "step_size_mm": 15, "direction": "right"},
            {"session_token": token, "step_size_mm": 10, "direction": "zero"},
            {"session_token": token, "step_size_mm": 10.0, "direction": "up"},
            {"session_token": token, "step_size_mm": 10, "direction": "right", "extra": 1},
        ):
            rejected = self.client.post("/api/action", json=bad)
            self.assertIn(rejected.status_code, (400, 422), rejected.text)
            self.assertNotIn("session_token", rejected.json())

        huge = self.client.post(
            "/api/action",
            content=b'{"session_token":"' + b"a" * 70000 + b'"}',
            headers={"content-type": "application/json"},
        )
        self.assertEqual(huge.status_code, 413, huge.text)

    def test_runtime_failure_has_no_fallback_token(self):
        start = self.client.post("/api/session", json={})
        token = start.json()["session_token"]
        with mock.patch.object(self.engine, "action_scene", side_effect=RuntimeError("forced")):
            failed = self.client.post("/api/action", json={
                "session_token": token, "step_size_mm": 10, "direction": "left",
            })
        self.assertEqual(failed.status_code, 503, failed.text)
        self.assertEqual(failed.json()["detail"], "Real model/physics runtime unavailable; no fallback")
        self.assertNotIn("session_token", failed.json())

    def test_restore_matches_continuous_scene_for_every_direction_and_size(self):
        _, mj, Env, _, _ = self.engine.get_backend()
        fresh = Env()
        try:
            fresh.reset()
            first = self.adapter.stride64(fresh.render())
            fresh.step((0.0, 0.0))
            mj.mj_forward(fresh.model, fresh.data)
            second = self.adapter.stride64(fresh.render())
            api_body, state, previous, current = self.engine.reset_scene()
            np.testing.assert_array_equal(_integration(mj, fresh), state)
            np.testing.assert_array_equal(first, previous)
            np.testing.assert_array_equal(second, current)
            self.assertEqual(api_body["scored_actions"], 0)
            self.assertAlmostEqual(api_body["physics_time_s"], 0.05, delta=1e-12)

            for index, (direction, mm) in enumerate(SEQUENCE, start=1):
                delta = tuple(axis * mm / 1000.0 for axis in OFFSETS[direction])
                direct = self.adapter.infer(
                    self.engine.get_backend()[3], self.engine.get_backend()[4],
                    previous, current, delta, self.engine.get_backend()[0],
                )
                fresh.step(delta)
                mj.mj_forward(fresh.model, fresh.data)
                observed = fresh.observe()
                nxt = self.adapter.stride64(fresh.render())
                served, new_state, new_prev, new_cur = self.engine.action_scene(
                    state, previous, current, direction, mm,
                )
                np.testing.assert_array_equal(_integration(mj, fresh), new_state)
                np.testing.assert_array_equal(current, new_prev)
                np.testing.assert_array_equal(nxt, new_cur)
                np.testing.assert_allclose(observed["block_xy"], served["actual"]["actual_xy_m"], atol=0, rtol=0)
                np.testing.assert_allclose(observed["pusher_xy"], served["actual"]["pusher_xy_m"], atol=0, rtol=0)
                self.assertAlmostEqual(observed["block_yaw"], served["actual"]["actual_yaw_rad"], delta=0)
                velocity = fresh.data.qvel[fresh._block_dof_address:fresh._block_dof_address + 6]
                np.testing.assert_array_equal(observed["block_linear_velocity"], tuple(float(v) for v in velocity[:3]))
                np.testing.assert_array_equal(observed["block_angular_velocity"], tuple(float(v) for v in velocity[3:]))
                self.assertEqual(observed["pusher_contact"], served["actual"]["actual_contact"])
                self.assertEqual(served["prediction"]["ic01_xy_m"], direct["ic01_xy_m"])
                self.assertEqual(served["prediction"]["m6_xy_m"], direct["m6_xy_m"])
                self.assertEqual(served["prediction"]["correction_m"], direct["correction_m"])
                self.assertEqual(served["ic01_forward_calls"], 1)
                self.assertEqual(served["physics_step_calls"], 1)
                self.assertAlmostEqual(served["physics_time_s"], 0.05 * (index + 1), delta=1e-12)
                correction = np.linalg.norm(served["prediction"]["correction_m"])
                self.assertLessEqual(float(correction), 0.004000000001)
                if not served["prediction"]["predicted_contact"]:
                    self.assertEqual(float(correction), 0.0)
                state, previous, current = new_state, new_prev, new_cur
        finally:
            fresh.close()

    def test_two_sessions_do_not_share_physics_results(self):
        _, state_a, prev_a, cur_a = self.engine.reset_scene()
        _, state_b, prev_b, cur_b = self.engine.reset_scene()
        np.testing.assert_array_equal(state_a, state_b)
        served_a, next_a, _, _ = self.engine.action_scene(state_a, prev_a, cur_a, "right", 20)
        served_b, next_b, _, _ = self.engine.action_scene(state_b, prev_b, cur_b, "up", 5)
        self.assertFalse(np.array_equal(next_a, next_b))
        self.assertNotEqual(served_a["actual"]["pusher_xy_m"], served_b["actual"]["pusher_xy_m"])
        self.assertNotEqual(served_a["real_rgb_png"], served_b["real_rgb_png"])


if __name__ == "__main__":
    unittest.main()
