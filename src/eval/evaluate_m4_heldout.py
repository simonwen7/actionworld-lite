"""ActionWorld Lite M4 / IC-01 single-step held-out evaluator, PRE-REGISTERED.

--self-test: pure synthetic NumPy metrics only; NEVER constructs any dataset.
--run-one-time-test: permitted ONLY in a later explicitly approved task.

One-step RGB/action-conditioned PushT research candidate; NOT a multi-step rollout.
All TEST outcomes, including failures to meet comparative bars, are reportable.
This program performs no training, calibration, threshold selection, model writes,
or Git operations. In TEST mode it reads TEST exactly once in deterministic order.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_HASHES = {
    "src/data/world_model_dataset.py": "0c30473110082e2f0bc851b8f03f9a4f7c9f13dec917cfb3606022aebce96ab3",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
    "src/models/visual_dynamics_v03.py": "d50c8debb7b04477861d248e4c6e5ec87ca63b247f48231bb4a9aa198619a659",
    "src/models/yaw_specialist_v04.py": "972b6067932a1b3de3f99f6dbb4d63129aae8c14daf24d4eacc9a6c6db35509c",
    "src/models/integrated_pusht_m4.py": "e6aeb125e4556ec0855cc77a528754c1655bcb2f1f0d71ff2b1a140bff5fc49a",
    "src/train/train_visual_dynamics.py": "b58eebfa61764dd98b6d6a743beadd6a2ae9ea6244e6dedfcc16620213d38edc",
    "src/train/train_visual_dynamics_v02.py": "e2c0f003ad956c39ba198f11b6fc5f5e06cf038e9e5a662d15a4eacd9e872963",
    "src/train/train_visual_dynamics_v03.py": "11e1c366d5b23df7abf6feda4f48bc0c2961c2c7221fcb49812ab5b8acdcf570",
    "src/train/train_yaw_specialist_v04.py": "bd3d47c943d66b9b2956c75e54409447085b1ff612e57ec0a70a8f5da84963fe",
    "data/raw/m3_main_v01/manifest.json": "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50",
    "data/processed/m3_main_v01/splits.json": "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa",
    "data/processed/m4_visual_v01/best_model.pt": "3af77d096be257f3b0f806474a3995ead252f4bc0fb25e4543f248dfd59926d5",
    "data/processed/m4_visual_v01/training_history.json": "234bf0a9b24acb15e44debb63a8a39c6abd1e7b3cf93bc1c90681c79256bdaf4",
    "data/processed/m4_visual_v02/best_model.pt": "2b0417fdefcc188b4852bbc7c472c3cf3136ddde00b70e621c3c074db2d65463",
    "data/processed/m4_visual_v02/training_history.json": "546184c1d0824075a7123c5f6b7993594afc90e7afc5c507e57f0932a1890ba2",
    "data/processed/m4_visual_v03/best_model.pt": "d0e24224680e557b2619595a3adcfc09183d38bc43a22cb699a074df5f8a95bd",
    "data/processed/m4_visual_v03/training_history.json": "6a6c0c80869c4b0166bff0717658e94a39629eb8f4fa71a90578920465886144",
    "data/processed/m4_yaw_v04/best_model.pt": "50cb95f37ed1b2a342e4ef669324518d45b5f6509be2fa53088f4ecf8a92d93e",
    "data/processed/m4_yaw_v04/training_history.json": "4f351a582e30ef8d223a20511ec03dc98ca4d6b599e6df725a8c5ee3240f83c1",
}
EXPECTED_COUNT = 1701
MODES = ("center_push", "offset_push", "free_motion")
CONTACT_CUTOFF_M = 0.0008668007749070838


def verify_19(stage):
    assert len(EXPECTED_HASHES) == 19
    for name, expected in EXPECTED_HASHES.items():
        p = Path(name)
        if not p.is_file():
            raise AssertionError((stage, "missing frozen file", name))
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError((stage, "frozen sha mismatch", name, actual))
    print(f"{stage}: ALL 19 FROZEN SHA256 CHECKS PASSED", flush=True)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected one-dimensional finite metric values")
    if len(values) == 0:
        return None
    return {"n": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(values.max())}


def angular_error_deg(pred_cs, target_cs):
    pred_cs = np.asarray(pred_cs, dtype=np.float64)
    target_cs = np.asarray(target_cs, dtype=np.float64)
    if pred_cs.shape != target_cs.shape or pred_cs.ndim != 2 or pred_cs.shape[1] != 2:
        raise ValueError("Yaw cos/sin shape mismatch")
    pp = np.arctan2(pred_cs[:, 1], pred_cs[:, 0])
    tt = np.arctan2(target_cs[:, 1], target_cs[:, 0])
    return np.rad2deg(np.abs(np.arctan2(np.sin(pp - tt), np.cos(pp - tt))))


def ranking_auc_ap(score, actual):
    """Rank ROC-AUC and average precision, tie-aware, no external packages."""
    score = np.asarray(score, dtype=np.float64)
    actual = np.asarray(actual, dtype=bool)
    if score.ndim != 1 or score.shape != actual.shape or not np.isfinite(score).all():
        raise ValueError("Invalid ranking input")
    n, np_, nn = len(score), int(actual.sum()), int((~actual).sum())
    if n == 0 or np_ == 0 or nn == 0:
        return {"roc_auc": None, "average_precision": None,
                "random_ap_reference": None}
    indices = np.argsort(-score, kind="mergesort")
    ordered = actual[indices]
    scores = score[indices]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]) + 1, n]
    tp = np.cumsum(ordered, dtype=np.int64)[ends - 1].astype(float)
    fp = ends - tp
    recalls = tp / np_
    precisions = tp / ends
    ap = float(np.sum(np.diff(np.r_[0., recalls]) * precisions))
    tpr = np.r_[0., recalls]
    fpr = np.r_[0., fp / nn]
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    if not (0 <= auc <= 1 and 0 <= ap <= 1):
        raise AssertionError("Out-of-range ranking result")
    return {"roc_auc": auc, "average_precision": ap,
            "random_ap_reference": np_ / n}


def contact_report(actual, predicted, score):
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.shape != score.shape:
        raise ValueError("Contact shape mismatch")
    n = len(actual)
    if n == 0:
        return None
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
    pos, neg = int(actual.sum()), int((~actual).sum())
    if tp + fp + tn + fn != n:
        raise AssertionError("Confusion matrix count mismatch")
    # undefined precision / recall reported null, not silently 0.
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / pos if pos else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if pos and 2 * tp + fp + fn else None
    return {"n": n, "positives": pos, "negatives": neg,
            "prevalence": pos / n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "specificity": tn / neg if neg else None,
            **ranking_auc_ap(score, actual)}


def self_test():
    # This function imports neither Torch nor the dataset loader; no dataset access.
    np.testing.assert_allclose(angular_error_deg([[1., 0.]], [[-1., 0.]]), [180.])
    np.testing.assert_allclose(angular_error_deg([[-1., 0.]], [[-1., -0.]]), [0.], atol=1e-12)
    assert summary([0, 1, 9, 10])["mean"] == 5.0
    assert summary([]) is None
    assert ranking_auc_ap([4, 3, 2, 1], [True, True, False, False])["roc_auc"] == 1.0
    assert ranking_auc_ap([4, 3, 2, 1], [False, False, True, True])["roc_auc"] == 0.0
    assert ranking_auc_ap([1, 1, 1, 1], [True, False, True, False])["roc_auc"] == 0.5
    assert ranking_auc_ap([1, 1, 1, 1], [True, False, True, False])["average_precision"] == 0.5
    assert ranking_auc_ap([1, 2], [False, False])["roc_auc"] is None
    r = contact_report([1, 1, 0, 0], [1, 0, 1, 0], [2, 1, 0, -1])
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 1, 1, 1)
    assert r["precision"] == r["recall"] == r["f1"] == .5
    assert contact_report([0, 0], [0, 0], [-1, -2])["recall"] is None
    print("M4 TASK 3AG SYNTHETIC METRIC SELF-TEST: PASSED", flush=True)
    print("NO real TRAIN, VALIDATION, or TEST dataset accessed", flush=True)


def run_one_time_test():
    # This branch MUST NEVER run in Task 3AG. A separate user approval is required.
    import torch
    from torch.utils.data import DataLoader
    from src.data.world_model_dataset import ActionWorldWindowDataset
    from src.models.visual_dynamics import VisualPhysicalWorldModel
    from src.models.visual_dynamics_v03 import GatedVisualPhysicalWorldModel
    from src.models.integrated_pusht_m4 import IntegratedPushTWorldModel, CONTACT_CUTOFF_M as MODULE_CUTOFF

    verify_19("BEFORE")
    if MODULE_CUTOFF != CONTACT_CUTOFF_M:
        raise AssertionError("Frozen contact cutoff mismatch")
    torch.set_num_threads(2)
    integrated = IntegratedPushTWorldModel.restore_frozen()
    old_models = {}
    for key, klass, ckptpath, num_parameters in (
        ("v02", VisualPhysicalWorldModel, "data/processed/m4_visual_v02/best_model.pt", 415896),
        ("v03", GatedVisualPhysicalWorldModel, "data/processed/m4_visual_v03/best_model.pt", 424345),
    ):
        model = klass().cpu().eval()
        checkpoint = torch.load(ckptpath, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(f"Bad frozen model checkpoint: {ckptpath}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != num_parameters:
            raise AssertionError((key, "parameter count mismatch"))
        model.requires_grad_(False)
        if not all(torch.isfinite(p).all().item() for p in model.state_dict().values()):
            raise AssertionError((key, "nonfinite frozen weight"))
        old_models[key] = model
    if integrated.training or any(p.requires_grad for p in integrated.parameters()):
        raise AssertionError("Integrated model not frozen")

    # FIRST and ONLY TEST construction is here, after all frozen checks/restores.
    ds = ActionWorldWindowDataset("test")
    if len(ds) != EXPECTED_COUNT:
        raise AssertionError(("unexpected TEST window count", len(ds)))
    keys = ("current_xy", "next_xy", "delta_xy", "current_yaw_cs", "next_yaw_cs")
    records = []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            frames = batch["model_input"]["frames"]
            action = batch["model_input"]["action"]
            n = len(frames)
            if frames.device.type != "cpu" or action.device.type != "cpu":
                raise AssertionError("Non-CPU inference")
            out = integrated(frames, action)
            older = {name: model(frames, action) for name, model in old_models.items()}
            if tuple(out["next_contact"].shape) != (n,) or out["next_contact"].dtype != torch.bool:
                raise AssertionError("Contact output shape/dtype mismatch")
            if tuple(out["contact_score_m"].shape) != (n,):
                raise AssertionError("Contact score shape mismatch")
            if not torch.equal(out["next_contact"], out["contact_score_m"] >= integrated.contact_cutoff_m):
                raise AssertionError("Contact rule drift")
            torch.testing.assert_close(out["next_xy"], out["current_xy"] + out["delta_xy"], atol=0, rtol=0)
            torch.testing.assert_close(out["delta_xy"][~out["next_contact"]],
                                       torch.zeros_like(out["delta_xy"][~out["next_contact"]]), atol=0, rtol=0)
            torch.testing.assert_close(out["delta_xy"][out["next_contact"]],
                                       out["raw_delta_xy"][out["next_contact"]], atol=0, rtol=0)
            torch.testing.assert_close(out["next_yaw_cs"].norm(dim=1),
                                       torch.ones(n, dtype=torch.float64), atol=1e-8, rtol=1e-8)
            for name in keys:
                if tuple(out[name].shape) != (n, 2):
                    raise AssertionError(("Bad integrated output shape", name))
            for name, old in older.items():
                for key in ("next_xy", "delta_xy", "next_yaw_cs"):
                    if tuple(old[key].shape) != (n, 2):
                        raise AssertionError((name, key, "bad legacy output shape"))
            labels = batch["labels"]
            for i in range(n):
                rec = {
                    "mode": str(batch["metadata"]["mode"][i]),
                    "contact": bool(labels["next_contact"][i].item() >= .5),
                    "true_cur": labels["current_xy"][i].numpy().astype(np.float64),
                    "true_next": labels["next_xy"][i].numpy().astype(np.float64),
                    "true_delta": labels["delta_xy"][i].numpy().astype(np.float64),
                    "true_cur_cs": labels["current_yaw_cs"][i].numpy().astype(np.float64),
                    "true_next_cs": labels["next_yaw_cs"][i].numpy().astype(np.float64),
                    "cur": out["current_xy"][i].numpy(),
                    "next": out["next_xy"][i].numpy(),
                    "delta": out["delta_xy"][i].numpy(),
                    "cur_cs": out["current_yaw_cs"][i].numpy(),
                    "next_cs": out["next_yaw_cs"][i].numpy(),
                    "pred_contact": bool(out["next_contact"][i].item()),
                    "score": float(out["contact_score_m"][i].item()),
                }
                for name, old in older.items():
                    rec[f"{name}_delta"] = old["delta_xy"][i].numpy().astype(np.float64)
                    rec[f"{name}_next_native"] = old["next_xy"][i].numpy().astype(np.float64)
                    rec[f"{name}_next_yaw_native"] = old["next_yaw_cs"][i].numpy().astype(np.float64)
                    prob = torch.sigmoid(old["next_contact_logits"][i].reshape(()))
                    rec[f"{name}_prob"] = float(prob.item())
                records.append(rec)
    if len(records) != EXPECTED_COUNT:
        raise AssertionError(("unexpected TEST evaluated", len(records)))
    def stack(key):
        arr = np.stack([r[key] for r in records])
        if not np.isfinite(arr).all():
            raise AssertionError(("nonfinite prediction or label", key))
        return arr
    mode = np.asarray([r["mode"] for r in records])
    counts = {name: int((mode == name).sum()) for name in MODES}
    if counts != {name: 567 for name in MODES}:
        raise AssertionError(("unexpected TEST mode counts", counts))
    actual_contact = np.asarray([r["contact"] for r in records], dtype=bool)
    predicted_contact = np.asarray([r["pred_contact"] for r in records], dtype=bool)
    score = np.asarray([r["score"] for r in records], dtype=np.float64)
    if not np.isfinite(score).all():
        raise AssertionError("nonfinite geometry contact scores")
    current_truth = stack("true_cur"); next_truth = stack("true_next")
    delta_truth = stack("true_delta")
    cur_yaw_truth = stack("true_cur_cs"); next_yaw_truth = stack("true_next_cs")
    current_pred = stack("cur"); next_pred = stack("next"); delta_pred = stack("delta")
    cur_yaw_pred = stack("cur_cs"); next_yaw_pred = stack("next_cs")
    moving = np.linalg.norm(delta_truth, axis=1) >= .001
    if not moving.any() or not (~moving).any():
        raise AssertionError("Expected both TEST moving and static windows")
    masks = {
        "overall": np.ones(EXPECTED_COUNT, dtype=bool),
        "center_push": mode == "center_push", "offset_push": mode == "offset_push",
        "free_motion": mode == "free_motion", "moving": moving,
        "static": ~moving, "contact": actual_contact, "no_contact": ~actual_contact,
    }
    integrated_metrics = {
        "current_xy_cm": 100. * np.linalg.norm(current_pred - current_truth, axis=1),
        "next_xy_cm": 100. * np.linalg.norm(next_pred - next_truth, axis=1),
        "motion_cm": 100. * np.linalg.norm(delta_pred - delta_truth, axis=1),
        "current_yaw_deg": angular_error_deg(cur_yaw_pred, cur_yaw_truth),
        "next_yaw_deg": angular_error_deg(next_yaw_pred, next_yaw_truth),
    }
    comparison_motion = {"zero": 100. * np.linalg.norm(delta_truth, axis=1)}
    comparison_next = {"zero": 100. * np.linalg.norm(current_pred - next_truth, axis=1)}
    legacy = {}
    for name in ("v02", "v03"):
        raw = stack(f"{name}_delta")
        comparison_motion[name] = 100. * np.linalg.norm(raw - delta_truth, axis=1)
        comparison_next[name] = 100. * np.linalg.norm(current_pred + raw - next_truth, axis=1)
        legacy[name] = {
            "native_next_xy_cm": 100. * np.linalg.norm(stack(f"{name}_next_native") - next_truth, axis=1),
            "native_next_yaw_deg": angular_error_deg(stack(f"{name}_next_yaw_native"), next_yaw_truth),
            "contact_probability": np.asarray([r[f"{name}_prob"] for r in records], dtype=np.float64),
        }
    checks = {
        "overall_motion_lt_zero": bool(integrated_metrics["motion_cm"].mean() < comparison_motion["zero"].mean()),
        "moving_motion_le_v02": bool(integrated_metrics["motion_cm"][moving].mean() <= comparison_motion["v02"][moving].mean()),
        "static_motion_le_v02": bool(integrated_metrics["motion_cm"][~moving].mean() <= comparison_motion["v02"][~moving].mean()),
        "reanchored_next_xy_le_v02_raw": bool(integrated_metrics["next_xy_cm"].mean() <= comparison_next["v02"].mean()),
    }
    output = {
        "experiment": "M4 IC-01 ONE-TIME HELD-OUT SINGLE-STEP TEST",
        "scope": {"n": EXPECTED_COUNT, "mode_counts": counts,
                  "true_contact": int(actual_contact.sum()), "true_moving": int(moving.sum()),
                  "contact_cutoff_m": CONTACT_CUTOFF_M,
                  "no_test_tuning": True, "rollout_evaluated": False},
        "cohorts": {
            name: {
                "count": int(mask.sum()),
                "errors": {metric: summary(values[mask]) for metric, values in integrated_metrics.items()},
                "contact": contact_report(actual_contact[mask], predicted_contact[mask], score[mask]),
            } for name, mask in masks.items()
        },
        "comparison": {
            "zero_motion": {
                "motion_cm": {name: summary(comparison_motion["zero"][mask]) for name, mask in masks.items()},
                "reanchored_next_xy_cm": {name: summary(comparison_next["zero"][mask]) for name, mask in masks.items()},
            },
            "baselines": {
                key: {
                    "motion_cm": {name: summary(comparison_motion[key][mask]) for name, mask in masks.items()},
                    "reanchored_next_xy_cm": {name: summary(comparison_next[key][mask]) for name, mask in masks.items()},
                    "native_next_xy_cm": {name: summary(legacy[key]["native_next_xy_cm"][mask]) for name, mask in masks.items()},
                    "native_next_yaw_deg": {name: summary(legacy[key]["native_next_yaw_deg"][mask]) for name, mask in masks.items()},
                    "contact_at_0_5": {name: contact_report(actual_contact[mask],
                                  legacy[key]["contact_probability"][mask] >= .5,
                                  legacy[key]["contact_probability"][mask]) for name, mask in masks.items()},
                } for key in ("v02", "v03")
            },
        },
        "four_predeclared_test_comparisons": checks,
        "all_four": bool(all(checks.values())),
    }
    # TEST acceptance is reported, never used to modify any decision threshold.
    print("M4 ONE-TIME HELD-OUT TEST REPORT BEGIN", flush=True)
    print(json.dumps(output, sort_keys=True, ensure_ascii=True, allow_nan=False), flush=True)
    print("M4 ONE-TIME HELD-OUT TEST REPORT END", flush=True)
    verify_19("AFTER")
    print("M4 FINAL TEST: EVALUATION COMPLETED; ALL RESULTS MUST BE RETAINED", flush=True)
    print("M4 FINAL TEST: 4-COMPARISON PASS =", bool(all(checks.values())), flush=True)


def main():
    parser = argparse.ArgumentParser(description="M4 sealed one-time held-out evaluator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run-one-time-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run_one_time_test:
        run_one_time_test()
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
