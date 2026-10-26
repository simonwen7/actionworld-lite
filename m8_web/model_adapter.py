"""Frozen M4 + fitted M6 inference adapter, never imports Tkinter or datasets."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    "src/models/integrated_pusht_m4.py": "e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a",
    "src/research/m6_residual_v01.py": "fb7362cb7cacea2c28c771ae64378df09deaf04862de78b72ebf1cadb89be586",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
    "src/models/visual_dynamics_v03.py": "d50c8debb7b04477861d248e4c6e5ec87ca63b247f48231bb4a9aa198619a659",
    "src/models/yaw_specialist_v04.py": "972b6067932a1b3de3f99f6dbb4d63129aae8c14daf24d4eacc9a6c6db35509c",
    "src/simulation/pusht_env.py": "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml": "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
    "data/processed/m4_visual_v03/best_model.pt": "d0e24224680e557b2619595a3adcfc09183d38bc43a22cb699a074df5f8a95bd",
    "data/processed/m4_yaw_v04/best_model.pt": "50cb95f37ed1b2a342e4ef669324518d45b5f6509be2fa53088f4ecf8a92d93e",
    "data/processed/m6_residual_v01/model.json": "7e2bcba8eb9f2ad24690149b590ccfebc12e02ee82ecca72409c45aefcd50f44",
}
FINGERPRINT = hashlib.sha256(json.dumps(ARTIFACTS, sort_keys=True).encode()).hexdigest()
DYNAMICS = "data/processed/m4_visual_v03/best_model.pt"
YAW = "data/processed/m4_yaw_v04/best_model.pt"
FIT = "data/processed/m6_residual_v01/model.json"


def assert_frozen_artifacts(root=ROOT):
    for relative, expected in ARTIFACTS.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen M8 runtime input: {relative}")
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError(f"Frozen SHA mismatch: {relative}")
    return FINGERPRINT


def load_fitted(root=ROOT):
    """Return validated M6 NumPy coefficients. Read-only; never fit anything."""
    from src.research.m6_residual_v01 import FEATURES, ABLATION_INDEX, CAP_M
    data = json.loads((Path(root) / FIT).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "actionworld-m6-contact-residual-v01":
        raise ValueError("Invalid M6 schema")
    if (data.get("source_sha256") != ARTIFACTS["src/research/m6_residual_v01.py"] or
        data.get("features") != list(FEATURES) or
        data.get("ablation_indices") != list(ABLATION_INDEX) or
        data.get("fit_only_predicted_contact") is not True or
        data.get("validation_not_used_for_selection") is not True or
        data.get("seed") != 20261026 or data.get("full_train_active_windows") != 2223 or
        data.get("correction_cap_m") != CAP_M):
        raise ValueError("Changed M6 model contract")
    ck = data.get("checkpoints_sha256")
    if not isinstance(ck, dict) or ck.get("dynamics") != ARTIFACTS[DYNAMICS] or ck.get("yaw") != ARTIFACTS[YAW]:
        raise ValueError("Incorrect M4 weights in fitted M6 metadata")
    selection = data.get("choice")
    if not isinstance(selection, dict) or selection.get("lambda") != 1.0 or selection.get("alpha") != 1.0 or abs(float(selection.get("score", float("nan"))) - 0.35388421624620825) > 1e-12:
        raise ValueError("M6 hyperparameter/provenance mismatch")

    def parse(value, n):
        if not isinstance(value, dict) or value.get("lambda") != 1.0:
            raise ValueError("Incorrect ridge parameters")
        result = {}
        for key, shape in (("mean", (n,)), ("std", (n,)), ("coef", (n + 1, 2))):
            x = value.get(key)
            if not isinstance(x, list) or any(isinstance(z, bool) for z in (x if key != "coef" else [v for row in x if isinstance(row,list) for v in row])):
                raise ValueError("Non numeric fitted parameter")
            a = np.asarray(x, dtype=np.float64)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Invalid ridge coefficient shape/finite values")
            result[key] = a
        if np.any(result["std"] <= 0):
            raise ValueError("Invalid ridge std")
        return result
    return {"full": parse(data.get("trained_full"), 10), "alpha": 1.0, "cap_m": CAP_M}


def stride64(rgb256):
    frame = np.asarray(rgb256)
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise ValueError("Invalid MuJoCo 256 RGB")
    return np.ascontiguousarray(frame[::4, ::4])


def input_tensors(previous, current, action_m, torch):
    prev, cur = np.asarray(previous), np.asarray(current)
    if prev.shape != (64,64,3) or cur.shape != (64,64,3) or prev.dtype != np.uint8 or cur.dtype != np.uint8:
        raise ValueError("Invalid real RGB pair")
    delta = np.asarray(action_m, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .02000000001:
        raise ValueError("Out-of-range physical action")
    frames = np.concatenate((prev.transpose(2,0,1), cur.transpose(2,0,1)), axis=0)
    return (torch.from_numpy(np.ascontiguousarray(frames)).unsqueeze(0).to(torch.float32).div_(255.0),
            torch.tensor([delta / .02], dtype=torch.float32))


def infer(model, fitted, previous, current, action_m, torch):
    from src.research.m6_residual_v01 import features_from_predictions, predict_ridge, bounded_contact_correction
    frames, action = input_tensors(previous, current, action_m, torch)
    with torch.inference_mode():
        output = model(frames, action)  # ONE genuine IC-01 forward
    out = {key: value.detach().cpu().numpy() for key,value in output.items()}
    feature, contact = features_from_predictions(action.numpy(), out)
    correction = bounded_contact_correction(predict_ridge(feature, fitted["full"]), contact, fitted["alpha"])
    original = out["next_xy"][0]
    corrected = (out["current_xy"] + out["delta_xy"] + correction)[0]
    delta_orig = out["delta_xy"][0]
    delta_fix = delta_orig + correction[0]
    yaw_cs = out["next_yaw_cs"][0]
    if not (np.isfinite(corrected).all() and np.linalg.norm(correction[0]) <= 0.004000000001):
        raise ValueError("Invalid bounded M6 correction")
    return {
        "ic01_xy_m": original.tolist(), "m6_xy_m": corrected.tolist(),
        "ic01_delta_m": delta_orig.tolist(), "m6_delta_m": delta_fix.tolist(),
        "ic01_yaw_rad": float(math.atan2(yaw_cs[1],yaw_cs[0])),
        "correction_m": correction[0].tolist(), "predicted_contact": bool(contact[0]),
        "contact_score_m": float(out["contact_score_m"][0]),
    }


def score_step(pred, before, after):
    actual_xy = np.asarray(after["block_xy"], dtype=np.float64)
    old_xy = np.asarray(before["block_xy"], dtype=np.float64)
    actual_delta = actual_xy - old_xy
    yaw_delta = math.atan2(math.sin(pred["ic01_yaw_rad"] - float(after["block_yaw"])),
                           math.cos(pred["ic01_yaw_rad"] - float(after["block_yaw"])))
    result = {
        "actual_xy_m": actual_xy.tolist(), "actual_yaw_rad": float(after["block_yaw"]),
        "pusher_xy_m": list(after["pusher_xy"]),
        "ic01_next_xy_error_cm": float(np.linalg.norm(np.asarray(pred["ic01_xy_m"])-actual_xy)*100),
        "m6_next_xy_error_cm": float(np.linalg.norm(np.asarray(pred["m6_xy_m"])-actual_xy)*100),
        "ic01_motion_error_cm": float(np.linalg.norm(np.asarray(pred["ic01_delta_m"])-actual_delta)*100),
        "m6_motion_error_cm": float(np.linalg.norm(np.asarray(pred["m6_delta_m"])-actual_delta)*100),
        "next_yaw_error_deg": float(abs(yaw_delta)*180/math.pi),
        "actual_contact": bool(after["pusher_contact"]),
        "contact_correct": bool(pred["predicted_contact"] == bool(after["pusher_contact"])),
    }
    if not all(math.isfinite(v) for v in result.values() if type(v) is float):
        raise ValueError("Nonfinite score")
    return result
