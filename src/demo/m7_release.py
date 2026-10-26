"""ActionWorld Lite M7: local-only MuJoCo / IC-01 / M6 live comparison.

One frozen IC-01 inference per scored action, TWO real RGB observations,
ONE MuJoCo physics step after prediction. No generated RGB or free rollout.

Modes:
  --self-test   Pure synthetic QA; no model weights, simulation or GUI.
  --run-demo    Requires locally held, exact-hash frozen artifacts.

The M4 one-time TEST and M6 train/validation experiment are never imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.demo.m5_interactive import (
    CANVAS_SIZE,
    IMAGE_SIZE,
    VIEW_HALF_EXTENT_M,
    box_outline,
    circular_yaw_error_deg,
    frame_to_model_rgb,
    make_model_inputs,
    ppm_bytes,
    world_to_canvas,
    yaw_of,
)
from src.research.m6_residual_v01 import (
    ABLATION_INDEX,
    CAP_M,
    FEATURES,
    bounded_contact_correction,
    features_from_predictions,
    predict_ridge,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DYNAMICS = "data/processed/m4_visual_v03/best_model.pt"
CHECKPOINT_YAW = "data/processed/m4_yaw_v04/best_model.pt"
MODEL_FILE = "data/processed/m6_residual_v01/model.json"
VALIDATION_FILE = "data/processed/m6_residual_v01/validation_report.json"
M6_SOURCE = "src/research/m6_residual_v01.py"
M5_SOURCE = "src/demo/m5_interactive.py"
M4_SOURCE = "src/models/integrated_pusht_m4.py"

LOCKED_SHA256 = {
    CHECKPOINT_DYNAMICS: "d0e24224680e557b2619595a3adcfc09183d38bc43a22cb699a074df5f8a95bd",
    CHECKPOINT_YAW: "50cb95f37ed1b2a342e4ef669324518d45b5f6509be2fa53088f4ecf8a92d93e",
    MODEL_FILE: "7e2bcba8eb9f2ad24690149b590ccfebc12e02ee82ecca72409c45aefcd50f44",
    VALIDATION_FILE: "e705812dac089e0c2722d389f37419318b7aab7c46cb7da2fe8ec2823e7f08b9",
    M6_SOURCE: "fb7362cb7cacea2c28c771ae64378df09deaf04862de78b72ebf1cadb89be586",
    M5_SOURCE: "5481176dabf76f28c2ad4411eadc698e81a4506af07d2b78abc5fc7c89fd0dac",
    M4_SOURCE: "e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a",
}

MODEL_SCHEMA = "actionworld-m6-contact-residual-v01"
STEP_MM = (5, 10, 20)
BG = "#F7F9FC"
INK = "#14243B"
ORANGE = "#D97736"
BLUE = "#0873C5"
PURPLE = "#7C3AED"
PUSHER = "#2B80B9"


def digest_file(path: Path) -> str:
    """Streaming digest: never parse sealed datasets or evaluators."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def check_local_artifacts(root: Path = ROOT) -> dict[str, str]:
    """Fail closed before model imports or simulator construction."""
    evidence = {}
    for relative, expected in LOCKED_SHA256.items():
        filename = root / relative
        if not filename.is_file():
            raise FileNotFoundError(f"Local release artifact missing: {relative}")
        actual = digest_file(filename)
        if actual != expected:
            raise RuntimeError(f"Immutable release artifact SHA256 mismatch: {relative}")
        evidence[relative] = actual
    return evidence


def require_number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a number")
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError(f"{description} must be finite")
    return answer


def parse_fitted_record(raw, dimension: int, description: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"{description} is not an object")
    result = {}
    for field, shape in (("mean", (dimension,)), ("std", (dimension,)),
                         ("coef", (dimension + 1, 2))):
        value = raw.get(field)
        # Reject non-numeric strings, bools, ragged arrays and nan/inf.
        if not isinstance(value, list):
            raise ValueError(f"{description}.{field} is not a list")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{description}.{field} invalid numeric values") from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{description}.{field} invalid shape/finite values")
        if array.dtype == np.bool_:
            raise ValueError(f"{description}.{field} is boolean")
        result[field] = array
    if np.any(result["std"] <= 0):
        raise ValueError(f"{description}.std must be positive")
    if require_number(raw.get("lambda"), description + ".lambda") != 1.0:
        raise ValueError(f"{description} uses unexpected fitted lambda")
    result["lambda"] = 1.0
    return result


def validate_fitted_model(data: dict) -> dict:
    """Validate the exact TRAIN-fitted M6 runtime contract, no fitting."""
    if not isinstance(data, dict) or data.get("schema") != MODEL_SCHEMA:
        raise ValueError("Unknown M6 model schema")
    if data.get("source_sha256") != LOCKED_SHA256[M6_SOURCE]:
        raise ValueError("M6 model/source mismatch")
    if data.get("features") != list(FEATURES):
        raise ValueError("M6 feature order mismatch")
    if data.get("ablation_indices") != list(ABLATION_INDEX):
        raise ValueError("M6 ablation feature mismatch")
    if data.get("fit_only_predicted_contact") is not True:
        raise ValueError("M6 contact fitting policy changed")
    if data.get("validation_not_used_for_selection") is not True:
        raise ValueError("M6 selection boundary changed")
    if require_number(data.get("correction_cap_m"), "cap") != CAP_M:
        raise ValueError("M6 4 mm bound changed")
    if data.get("seed") != 20261026 or data.get("full_train_active_windows") != 2223:
        raise ValueError("M6 training provenance mismatch")
    checkpoints = data.get("checkpoints_sha256")
    if not isinstance(checkpoints, dict) or checkpoints.get("dynamics") != LOCKED_SHA256[CHECKPOINT_DYNAMICS] or checkpoints.get("yaw") != LOCKED_SHA256[CHECKPOINT_YAW]:
        raise ValueError("M6 fitted model refers to different M4 checkpoints")
    choice = data.get("choice")
    if not isinstance(choice, dict) or require_number(choice.get("alpha"), "alpha") != 1.0 or require_number(choice.get("lambda"), "lambda") != 1.0:
        raise ValueError("M6 selected hyperparameters mismatch")
    if abs(require_number(choice.get("score"), "score") - 0.35388421624620825) > 1e-12:
        raise ValueError("M6 internal model choice score changed")
    full = parse_fitted_record(data.get("trained_full"), len(FEATURES), "trained_full")
    ablation = parse_fitted_record(data.get("trained_ablation"), len(ABLATION_INDEX), "trained_ablation")
    const = np.asarray(data.get("constant_residual_m"), dtype=np.float64)
    if const.shape != (2,) or not np.isfinite(const).all():
        raise ValueError("M6 constant baseline invalid")
    return {"full": full, "ablation": ablation, "alpha": 1.0,
            "cap_m": CAP_M, "features": FEATURES}


def load_accepted_model(root: Path = ROOT) -> dict:
    """Called only after complete artifact integrity validation."""
    data = json.loads((root / MODEL_FILE).read_text(encoding="utf-8"))
    return validate_fitted_model(data)


def as_numpy(value) -> np.ndarray:
    """Convert frozen inference tensor to a detached CPU array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compare_one_action(normalized_action: np.ndarray, frozen_output: dict,
                       fitted: dict) -> dict:
    """Predict two XY poses from ONE common IC-01 output, without labels.

    Accepts only present-time IC-01 output. The real post-step observation is
    deliberately not a parameter, so future-state leakage is impossible here.
    """
    if normalized_action.shape != (1, 2) or normalized_action.dtype != np.float32:
        raise ValueError("Expected single normalized float32 action")
    required = ("current_xy", "next_xy", "delta_xy", "next_yaw_cs",
                "current_yaw_cs", "pusher_xy", "next_contact", "contact_score_m")
    if any(name not in frozen_output for name in required):
        raise ValueError("Incomplete frozen IC-01 output")
    raw = {name: as_numpy(frozen_output[name]) for name in required}
    for name, shape in (("current_xy", (1, 2)), ("next_xy", (1, 2)),
                        ("delta_xy", (1, 2)), ("next_yaw_cs", (1, 2)),
                        ("current_yaw_cs", (1, 2)), ("pusher_xy", (1, 2)),
                        ("next_contact", (1,)), ("contact_score_m", (1,))):
        if raw[name].shape != shape:
            raise ValueError(f"Incorrect IC-01 output shape: {name}")
    if raw["next_contact"].dtype != np.bool_:
        raise ValueError("IC-01 contact output is not boolean")
    for name in required:
        if name != "next_contact" and not np.isfinite(raw[name]).all():
            raise ValueError(f"Nonfinite IC-01 output: {name}")
    if not np.allclose(raw["next_xy"], raw["current_xy"] + raw["delta_xy"],
                       rtol=0, atol=1e-8):
        raise ValueError("Inconsistent IC-01 next XY/delta")
    feat, active = features_from_predictions(normalized_action, raw)
    correction = bounded_contact_correction(
        predict_ridge(feat, fitted["full"]), active, fitted["alpha"])
    if correction.shape != (1, 2) or not np.isfinite(correction).all() or np.linalg.norm(correction[0]) > CAP_M + 1e-12:
        raise ValueError("M6 correction is invalid or exceeds 4 mm")
    baseline_xy = raw["next_xy"][0].astype(np.float64)
    residual_xy = (raw["current_xy"] + raw["delta_xy"] + correction)[0]
    yaw_cs = raw["next_yaw_cs"][0].astype(np.float64)
    if abs(float(np.linalg.norm(yaw_cs)) - 1) > 1e-5:
        raise ValueError("Frozen yaw unit vector invalid")
    return {
        "ic01_xy": tuple(float(x) for x in baseline_xy),
        "m6_xy": tuple(float(x) for x in residual_xy),
        "ic01_delta": tuple(float(x) for x in raw["delta_xy"][0]),
        "m6_delta": tuple(float(x) for x in raw["delta_xy"][0] + correction[0]),
        "yaw_rad": yaw_of(tuple(float(x) for x in yaw_cs)),
        "contact_pred": bool(active[0]),
        "contact_score_m": float(raw["contact_score_m"][0]),
        "correction_m": tuple(float(x) for x in correction[0]),
    }


def measure_predictions(pred: dict, before: dict, after: dict) -> dict:
    """Post-step scoring only; neither model has access to 'after'."""
    truth = tuple(float(x) for x in after["block_xy"])
    before_xy = tuple(float(x) for x in before["block_xy"])
    true_delta = (truth[0] - before_xy[0], truth[1] - before_xy[1])
    reality_yaw = float(after["block_yaw"])
    numbers = {
        "ic01_xy_cm": 100 * math.dist(pred["ic01_xy"], truth),
        "m6_xy_cm": 100 * math.dist(pred["m6_xy"], truth),
        "ic01_motion_cm": 100 * math.dist(pred["ic01_delta"], true_delta),
        "m6_motion_cm": 100 * math.dist(pred["m6_delta"], true_delta),
        "yaw_error_deg": circular_yaw_error_deg(pred["yaw_rad"], reality_yaw),
        "actual_xy_m": truth,
        "actual_yaw_rad": reality_yaw,
        "actual_contact": bool(after["pusher_contact"]),
        "pred_contact_correct": pred["contact_pred"] == bool(after["pusher_contact"]),
    }
    if not all(math.isfinite(float(v)) for v in numbers.values() if isinstance(v, (float, int))):
        raise ValueError("Nonfinite post-action metric")
    return numbers


class ActionCounter:
    """One successful physical action -> one scored action; reset excludes startup."""

    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def scored_step(self):
        self.count += 1
        return self.count


class SessionRecorder:
    """Append-only evidence outside repo; errors are raised, not silenced."""

    def __init__(self, root: Path | None = None):
        if root is None:
            root = Path.home() / "Desktop" / "ActionWorldLite_M7_Sessions"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = root / ("session_" + stamp + "_" + uuid4().hex[:8] + ".jsonl")
        self._f = self.path.open("x", encoding="utf-8")
        self.closed = False

    def write(self, event: str, **payload):
        if self.closed:
            raise RuntimeError("Session record already closed")
        record = {"event": event, "time_utc": datetime.now(timezone.utc).isoformat(), **payload}
        self._f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        self._f.flush()

    def close(self):
        if not self.closed:
            self._f.close()
            self.closed = True


def synthetic_self_test():
    """Pure synthetic tests. No checkpoints, datasets, MuJoCo or Tk window."""
    from src.research.m6_residual_v01 import FEATURES as M6_FEATURES

    assert tuple(M6_FEATURES) == FEATURES
    assert len(LOCKED_SHA256) == 7 and all(len(s) == 64 for s in LOCKED_SHA256.values())
    assert len(FEATURES) == 10
    assert len(ABLATION_INDEX) == 6
    counter = ActionCounter()
    assert counter.count == 0
    counter.scored_step()
    counter.scored_step()
    assert counter.count == 2
    counter.reset()
    assert counter.count == 0

    model = {
        "schema": MODEL_SCHEMA, "source_sha256": LOCKED_SHA256[M6_SOURCE],
        "features": list(FEATURES), "ablation_indices": list(ABLATION_INDEX),
        "fit_only_predicted_contact": True, "validation_not_used_for_selection": True,
        "correction_cap_m": 0.004, "seed": 20261026,
        "full_train_active_windows": 2223,
        "checkpoints_sha256": {"dynamics": LOCKED_SHA256[CHECKPOINT_DYNAMICS],
                                "yaw": LOCKED_SHA256[CHECKPOINT_YAW]},
        "choice": {"alpha": 1.0, "lambda": 1.0, "score": 0.35388421624620825},
        "trained_full": {"mean": [0.0] * 10, "std": [1.0] * 10,
                         "coef": [[0.0, 0.0]] * 11, "lambda": 1.0},
        "trained_ablation": {"mean": [0.0] * 6, "std": [1.0] * 6,
                            "coef": [[0.0, 0.0]] * 7, "lambda": 1.0},
        "constant_residual_m": [0.0, 0.0],
    }
    fitted = validate_fitted_model(model)
    action = np.array([[0.5, 0.0]], dtype=np.float32)
    output = {
        "current_xy": np.array([[0.0, 0.0]]),
        "next_xy": np.array([[0.001, 0.0]]),
        "delta_xy": np.array([[0.001, 0.0]]),
        "next_yaw_cs": np.array([[1.0, 0.0]]),
        "current_yaw_cs": np.array([[1.0, 0.0]]),
        "pusher_xy": np.array([[0.13, 0.0]]),
        "next_contact": np.array([True]),
        "contact_score_m": np.array([0.002]),
    }
    baseline = compare_one_action(action, output, fitted)
    assert baseline["m6_xy"] == baseline["ic01_xy"]
    # Intercept in fit is in normalized meters (the ridge function multiplies 0.02).
    fitted["full"]["coef"][0, 0] = 0.5  # would predict 10mm, capped at 4mm
    changed = compare_one_action(action, output, fitted)
    assert np.isclose(changed["correction_m"][0], 0.004, atol=1e-12)
    assert np.isclose(changed["m6_xy"][0] - changed["ic01_xy"][0], 0.004)
    no_contact = dict(output, next_contact=np.array([False]))
    result = compare_one_action(action, no_contact, fitted)
    assert result["m6_xy"] == result["ic01_xy"]
    fitted["alpha"] = 0.0
    result = compare_one_action(action, output, fitted)
    assert result["m6_xy"] == result["ic01_xy"]
    fitted["alpha"] = 1.0
    fake_before = {"block_xy": (0.0, 0.0)}
    fake_after = {"block_xy": (0.005, 0.0), "block_yaw": 0.0, "pusher_contact": True}
    score = measure_predictions(changed, fake_before, fake_after)
    assert np.isclose(score["ic01_xy_cm"], 0.4)
    assert np.isclose(score["m6_xy_cm"], 0.0)
    assert np.isclose(score["ic01_motion_cm"], 0.4)
    assert score["m6_motion_cm"] < 1e-10
    assert score["pred_contact_correct"]
    assert circular_yaw_error_deg(math.pi - .01, -math.pi + .01) < 2.0
    assert world_to_canvas(0, 0) == (CANVAS_SIZE / 2, CANVAS_SIZE / 2)
    assert len(box_outline((0.0, 0.0), 0.0, .045, .095, .025)) == 8
    assert len(ppm_bytes(np.zeros((256,256,3),dtype=np.uint8))) > 256*256*3
    assert frame_to_model_rgb(np.zeros((256,256,3),dtype=np.uint8)).shape == (64,64,3)
    assert make_model_inputs(np.zeros((64,64,3),dtype=np.uint8),
                             np.zeros((64,64,3),dtype=np.uint8),
                             (.01,0))[0].shape == (1,6,64,64)
    rejected = dict(model, features=list(reversed(FEATURES)))
    try:
        validate_fitted_model(rejected)
    except ValueError:
        pass
    else:
        raise AssertionError("Feature permutation was accepted")
    corrupt = dict(model, trained_full={**model["trained_full"], "coef": [[0,0]] * 10})
    try:
        validate_fitted_model(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong coefficient shape accepted")
    # Exercise the REAL GUI callbacks with fake torch, model, env and variables.
    # No real window or MuJoCo engine is ever instantiated.
    from contextlib import nullcontext
    from tempfile import TemporaryDirectory

    class FakeVar:
        def __init__(self, value=""):
            self.value = value
        def get(self):
            return self.value
        def set(self, value):
            self.value = value

    class FakeEnv:
        def __init__(self):
            self.steps = 0
            self.closes = 0
            self.xy = [0.0, 0.0]
        def reset(self):
            self.steps = 0
            self.xy = [0.0, 0.0]
        def render(self):
            return np.zeros((256, 256, 3), dtype=np.uint8)
        def step(self, action):
            self.steps += 1
            self.xy[0] += float(action[0]) * 0.5
            self.xy[1] += float(action[1]) * 0.5
            return self.observe()
        def observe(self):
            return {"time": self.steps * .05, "block_xy": tuple(self.xy),
                    "block_yaw": 0.0, "pusher_xy": (0.15, 0.0),
                    "pusher_contact": True}
        def close(self):
            self.closes += 1

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()
        @staticmethod
        def from_numpy(value):
            return value

    class FakeRoot:
        def __init__(self):
            self.destroyed = False
        def destroy(self):
            self.destroyed = True

    class FakeModel:
        def __init__(self):
            self.calls = 0
        def __call__(self, frames, action):
            self.calls += 1
            assert frames.shape == (1, 6, 64, 64)
            assert action.shape == (1, 2)
            return output

    with TemporaryDirectory(prefix="m7-synthetic-outside-repo-") as temporary:
        recorder = SessionRecorder(Path(temporary))
        fake = ReleaseDemo.__new__(ReleaseDemo)
        fake.env = FakeEnv()
        fake.model = FakeModel()
        fake.fitted = fitted
        fake.torch = FakeTorch()
        fake.recorder = recorder
        fake.root = FakeRoot()
        fake.previous = None
        fake.current = None
        fake.observation = None
        fake.last_pred = None
        fake.counter = ActionCounter()
        fake.busy = False
        fake.failed = False
        fake.closed = False
        fake.mm = FakeVar("10")
        fake.status, fake.metrics = FakeVar(), FakeVar()
        fake._busy = lambda value: setattr(fake, "busy", value)
        fake._render_frame = lambda rgb: None
        fake._render_world = lambda: None
        fake.reset()
        assert fake.counter.count == 0 and fake.env.steps == 1
        assert fake.model.calls == 0
        fake.act((1, 0))
        assert fake.model.calls == 1 and fake.env.steps == 2
        assert fake.counter.count == 1
        fake.act((0, 1))
        assert fake.model.calls == 2 and fake.env.steps == 3
        assert fake.counter.count == 2
        fake.reset()
        assert fake.counter.count == 0 and fake.env.steps == 1
        fake.close()
        assert fake.env.closes == 1 and fake.root.destroyed
        assert recorder.closed
        events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        assert [e["event"] for e in events] == [
            "reset", "action", "action", "reset", "close"
        ]
        assert [e["scored_actions"] for e in events] == [0, 1, 2, 0, 0]
        assert all(e["ic01_forward_calls"] == 1 and e["physics_step_calls"] == 1
                   for e in events if e["event"] == "action")

    # Invalid artifact fails before any weights are imported.
    try:
        check_local_artifacts(Path("/nonexistent-actionworld-m7-synthetic-root"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing release artifacts did not fail closed")
    print("M7 SYNTHETIC-ONLY SELF-TEST: PASSED")
    print("NO DATASET, TEST, MODEL WEIGHTS, LIVE INFERENCE, MUJOCO OR TK ROOT OPENED")


class ReleaseDemo:
    """GUI displays actual, frozen, corrected poses for one single-step action."""

    def __init__(self, root, env, model, fitted, torch_module, recorder):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root, self.env, self.model = root, env, model
        self.fitted, self.torch, self.recorder = fitted, torch_module, recorder
        self.previous = None
        self.current = None
        self.observation = None
        self.last_pred = None
        self.counter = ActionCounter()
        self.busy = False
        self.failed = False
        self.closed = False
        self._photo = None

        root.title("ActionWorld Lite | M7 Local Research Release")
        root.geometry("1140x795")
        root.minsize(1050, 730)
        root.configure(background=BG)
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="ActionWorld Lite", font=("Helvetica", 23, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Real physical interaction  /  Frozen IC-01  /  M6 bounded residual",
                  font=("Helvetica", 12)).pack(anchor="w", pady=(2, 3))
        ttk.Label(outer, text=("One-step, conditioned on two real RGB observations and the chosen action. "
                              "The learned model does NOT generate the RGB frame or run a free rollout."),
                  wraplength=1090).pack(anchor="w", pady=(0, 13))

        area = ttk.Frame(outer)
        area.pack(fill=tk.X)
        panel_rgb = ttk.LabelFrame(area, text="Actual MuJoCo RGB", padding=8)
        panel_rgb.grid(row=0, column=0, sticky="n", padx=(0, 12))
        self.real_image = ttk.Label(panel_rgb)
        self.real_image.pack()
        ttk.Label(panel_rgb, text="Live rendered physical scene, 256 × 256 RGB",
                  wraplength=260).pack(pady=(8,0))
        panel_world = ttk.LabelFrame(area, text="XY world coordinates (meters)", padding=8)
        panel_world.grid(row=0, column=1, sticky="n", padx=(0, 12))
        self.canvas = tk.Canvas(panel_world, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#FBFCFE", highlightthickness=0)
        self.canvas.pack()
        ttk.Label(panel_world, text="Orange actual  •  Blue IC-01  •  Purple M6",
                  wraplength=345).pack(pady=(8,0))
        panel_info = ttk.LabelFrame(area, text="Same-action comparison", padding=11)
        panel_info.grid(row=0, column=2, sticky="nsew")
        area.columnconfigure(2, weight=1)
        self.metrics = tk.StringVar(value="No action evaluated yet.")
        ttk.Label(panel_info, textvariable=self.metrics, justify="left",
                  font=("Menlo", 10), wraplength=310).pack(anchor="nw")

        controls = ttk.LabelFrame(outer, text="Manual pusher controls (one click = one 0.05 s physics step)", padding=10)
        controls.pack(fill=tk.X, pady=(14, 8))
        self.mm = tk.StringVar(value="10")
        ttk.Label(controls, text="Action size:").grid(row=0, column=0, padx=5)
        for col, mm in enumerate(STEP_MM, start=1):
            ttk.Radiobutton(controls, text=f"{mm} mm", variable=self.mm,
                            value=str(mm)).grid(row=0, column=col, padx=6)
        self.buttons = []
        for col, title, vector in ((1, "Left", (-1, 0)), (2, "Up", (0, 1)),
                                   (3, "Down", (0, -1)), (4, "Right", (1, 0)),
                                   (5, "Zero", (0, 0))):
            button = ttk.Button(controls, text=title, command=lambda d=vector: self.act(d))
            button.grid(row=1, column=col, padx=5, pady=7, sticky="ew")
            self.buttons.append(button)
        self.reset_button = ttk.Button(controls, text="Reset", command=self.reset)
        self.reset_button.grid(row=1, column=0, padx=5, pady=7)
        ttk.Label(controls, text="Use deliberate mouse clicks (keyboard auto-repeat disabled for reproducible counting).",
                  wraplength=1000).grid(row=2, column=0, columnspan=6, sticky="w")

        note = ttk.LabelFrame(outer, text="Research findings (unchanged M6 VALIDATION; not new TEST)", padding=9)
        note.pack(fill=tk.X, pady=(4, 6))
        ttk.Label(note, text=("Mean motion error: IC-01 0.156625 cm → M6 0.135983 cm (13.18% lower); "
                              "paired 27-episode bootstrap 95% CI for difference: [-0.031218, -0.011330] cm. "
                              "Static and no-contact mean errors increased. Fixed simulated PushT only; "
                              "M4 TEST remains sealed."), wraplength=1090).pack(anchor="w")
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(outer, textvariable=self.status, wraplength=1080).pack(anchor="w", pady=(6, 3))
        ttk.Label(outer, text=f"Local evidence: {self.recorder.path}",
                  wraplength=1080).pack(anchor="w")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset()

    def _busy(self, value: bool):
        self.busy = value
        state = self.tk.DISABLED if value or self.failed else self.tk.NORMAL
        for item in self.buttons:
            item.configure(state=state)
        self.reset_button.configure(state=self.tk.DISABLED if value else self.tk.NORMAL)

    def _render_frame(self, rgb):
        self._photo = self.tk.PhotoImage(data=ppm_bytes(rgb), format="PPM")
        self.real_image.configure(image=self._photo)

    def _outline(self, xy, yaw, color, fill="", dash=None):
        for center_y, half_x, half_y in ((.045, .095, .025), (-.020, .025, .060)):
            options = {"dash": dash} if dash else {}
            self.canvas.create_polygon(box_outline(xy, yaw, center_y, half_x, half_y),
                                       outline=color, fill=fill, width=2, **options)

    def _render_world(self):
        c = self.canvas
        c.delete("all")
        mid = CANVAS_SIZE / 2
        for k in range(0, CANVAS_SIZE+1, 30):
            c.create_line(k, 0, k, CANVAS_SIZE, fill="#E8EDF5")
            c.create_line(0, k, CANVAS_SIZE, k, fill="#E8EDF5")
        c.create_line(mid, 0, mid, CANVAS_SIZE, fill="#A3B4C9")
        c.create_line(0, mid, CANVAS_SIZE, mid, fill="#A3B4C9")
        if self.observation is None:
            return
        obs = self.observation
        self._outline(tuple(obs["block_xy"]), float(obs["block_yaw"]), ORANGE, "#F7C196")
        x, y = world_to_canvas(*obs["pusher_xy"])
        radius = 0.025 * CANVAS_SIZE / (2 * VIEW_HALF_EXTENT_M)
        c.create_oval(x-radius, y-radius, x+radius, y+radius,
                     fill=PUSHER, outline="#18568B")
        if self.last_pred is not None:
            p = self.last_pred
            self._outline(p["ic01_xy"], p["yaw_rad"], BLUE, dash=(7, 4))
            self._outline(p["m6_xy"], p["yaw_rad"], PURPLE, dash=(2, 2))
        c.create_text(9, 9, text="+Y ↑      +X →", anchor="nw", fill=INK)
        c.create_text(9, CANVAS_SIZE-9, text="Physics XY — not RGB pixel alignment",
                      anchor="sw", fill="#526579")

    def reset(self):
        if self.busy or self.closed:
            return
        self._busy(True)
        try:
            self.env.reset()
            first = self.env.render()
            previous = frame_to_model_rgb(first)
            self.env.step((0.0, 0.0))  # Required initial second real frame; unscored.
            after = self.env.observe()
            second = self.env.render()
            current = frame_to_model_rgb(second)
            self.previous, self.current = previous, current
            self.observation = after
            self.last_pred = None
            self.counter.reset()
            self.failed = False
            self._render_frame(second)
            self._render_world()
            self.metrics.set("Ready.\n0 scored actions.\nStartup 0 mm action excluded.\nBoth predictions await a click.")
            self.status.set("READY — uses prior genuine MuJoCo frames; click once per desired physics step.")
            self.recorder.write("reset", scored_actions=0, startup_zero_action_unscored=True,
                                physics_time_s=float(after["time"]))
        except Exception as exc:
            self.failed = True
            self.status.set(f"RESET FAILED — {type(exc).__name__}: {exc}")
            self.recorder.write("reset_failed", error_type=type(exc).__name__, message=str(exc))
        finally:
            self._busy(False)

    def act(self, direction):
        if self.busy or self.failed or self.closed or self.previous is None or self.current is None:
            return
        try:
            mm = int(self.mm.get())
            if mm not in STEP_MM or direction not in ((-1,0),(1,0),(0,-1),(0,1),(0,0)):
                raise ValueError("Invalid manual action")
            action_m = tuple(d * (mm / 1000.0) for d in direction)
            rgb_pair, action_normalized = make_model_inputs(self.previous, self.current, action_m)
        except Exception as exc:
            self.status.set(f"INVALID ACTION — {type(exc).__name__}: {exc}")
            self.recorder.write("invalid_action", error_type=type(exc).__name__, message=str(exc))
            return
        self._busy(True)
        try:
            prior = self.observation
            if prior is None:
                raise RuntimeError("Missing genuine pre-step MuJoCo state for scoring")
            with self.torch.inference_mode():
                output = self.model(self.torch.from_numpy(rgb_pair),
                                    self.torch.from_numpy(action_normalized))
            # Original IC-01 is invoked exactly once. No future state yet exists.
            pred = compare_one_action(action_normalized, output, self.fitted)
            real = self.env.step(action_m)  # ONE physical step after both predictions.
            new_rgb = self.env.render()
            next_low = frame_to_model_rgb(new_rgb)
            score = measure_predictions(pred, prior, real)
            # Commit observed state/counter only after complete post-step metrics.
            self.previous, self.current = self.current, next_low
            self.observation, self.last_pred = real, pred
            count = self.counter.scored_step()
            self._render_frame(new_rgb)
            self._render_world()
            self.metrics.set(
                f"Scored actions: {count}\nTime: {real['time']:.3f} s\n\n"
                f"Next XY error, IC-01: {score['ic01_xy_cm']:.3f} cm\n"
                f"Next XY error, M6:    {score['m6_xy_cm']:.3f} cm\n"
                f"Motion error, IC-01:  {score['ic01_motion_cm']:.3f} cm\n"
                f"Motion error, M6:     {score['m6_motion_cm']:.3f} cm\n"
                f"Yaw error (unchanged): {score['yaw_error_deg']:.3f}°\n\n"
                f"IC-01 XY: ({pred['ic01_xy'][0]:+.3f}, {pred['ic01_xy'][1]:+.3f}) m\n"
                f"M6 XY:    ({pred['m6_xy'][0]:+.3f}, {pred['m6_xy'][1]:+.3f}) m\n"
                f"Actual XY: ({score['actual_xy_m'][0]:+.3f}, {score['actual_xy_m'][1]:+.3f}) m\n\n"
                f"Contact: pred {pred['contact_pred']} / actual {score['actual_contact']}\n"
                f"Correct: {score['pred_contact_correct']}\n"
                f"M6 correction: {1000 * math.hypot(*pred['correction_m']):.3f} mm\n"
                f"Geometry score: {pred['contact_score_m']:+.5f} m"
            )
            self.recorder.write("action", scored_actions=count, step_size_mm=mm,
                                direction=list(direction), action_m=list(action_m),
                                physics_time_s=float(real["time"]),
                                prediction_source="two real RGB before action",
                                ic01_forward_calls=1, physics_step_calls=1,
                                ic01_xy_m=list(pred["ic01_xy"]), m6_xy_m=list(pred["m6_xy"]),
                                actual_xy_m=list(score["actual_xy_m"]),
                                correction_m=list(pred["correction_m"]),
                                ic01_motion_error_cm=score["ic01_motion_cm"],
                                m6_motion_error_cm=score["m6_motion_cm"],
                                ic01_next_xy_error_cm=score["ic01_xy_cm"],
                                m6_next_xy_error_cm=score["m6_xy_cm"],
                                next_yaw_error_deg=score["yaw_error_deg"],
                                predicted_contact=pred["contact_pred"],
                                actual_contact=score["actual_contact"],
                                contact_correct=score["pred_contact_correct"])
            self.status.set("OK — same real action, one IC-01 inference, two state predictions. NOT free-running.")
        except Exception as exc:
            self.failed = True
            self.status.set(f"ACTION FAILED — {type(exc).__name__}: {exc}. Reset required; no retry.")
            self.recorder.write("action_failed", error_type=type(exc).__name__, message=str(exc),
                                last_scored_actions=self.counter.count)
        finally:
            self._busy(False)

    def close(self):
        if self.closed:
            return
        if self.busy:
            self.root.after(100, self.close)
            return
        self.closed = True
        try:
            self.recorder.write("close", scored_actions=self.counter.count,
                                functional_failure=self.failed)
        finally:
            try:
                self.env.close()
            finally:
                self.recorder.close()
                self.root.destroy()


def run_demo():
    """Real demo requires accepted local files and explicit human permission."""
    evidence = check_local_artifacts()
    fitted = load_accepted_model()
    # If baseline metadata and parameter validation are wrong, stop before GUI.
    import torch
    from src.models.integrated_pusht_m4 import IntegratedPushTWorldModel
    from src.simulation.pusht_env import PushTEnv
    import tkinter as tk

    if torch.backends.mps.is_available():
        # MPS may be available, but this release remains explicitly CPU-only.
        pass
    frozen = IntegratedPushTWorldModel.restore_frozen(
        dynamics_checkpoint=ROOT / CHECKPOINT_DYNAMICS,
        yaw_checkpoint=ROOT / CHECKPOINT_YAW,
    )
    frozen.eval()
    frozen.requires_grad_(False)
    if any(p.requires_grad for p in frozen.parameters()):
        raise RuntimeError("Frozen IC-01 parameters unexpectedly trainable")
    recorder = SessionRecorder()
    recorder.write("session_start", schema="actionworld-m7-local-v01",
                   model_schema=MODEL_SCHEMA, artifact_sha256=evidence,
                   mode="CPU, local, two REAL RGB frames, one-step")
    env = None
    root = None
    try:
        env = PushTEnv()
        root = tk.Tk()
        _app = ReleaseDemo(root, env, frozen, fitted, torch, recorder)
        root.mainloop()
    except Exception as exc:
        if not recorder.closed:
            recorder.write("session_failure", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if env is not None:
            env.close()
        if root is not None:
            try:
                if root.winfo_exists():
                    root.destroy()
            except tk.TclError:
                pass
        recorder.close()
    print("M7 LOCAL DEMO CLOSED; SESSION_LOG:", recorder.path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--self-test", action="store_true")
    choice.add_argument("--run-demo", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
    else:
        run_demo()


if __name__ == "__main__":
    main()
