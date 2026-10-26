"""M6: frozen-IC-01, TRAIN-fitted contact-gated XY residual research.

Standalone CPU experiment. No TEST split or held-out evaluator is imported.
Hypothesis: a bounded, context-conditioned residual lowers validation motion
error versus the frozen predictor and a TRAIN-mean residual comparator.
Failure to improve is a valid negative result, not permission to retune VAL.
"""

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/processed/m6_residual_v01"
SEED = 20261026
SCALE_M = 0.02
CAP_M = 0.004
MOVING_M = 0.001
LAMBDAS = (1.0, 10.0, 100.0)
ALPHAS = (0.25, 0.5, 1.0)
FEATURES = (
    "action_x", "action_y", "action_norm", "relative_pusher_x",
    "relative_pusher_y", "current_yaw_cos", "current_yaw_sin",
    "predicted_delta_x", "predicted_delta_y", "contact_score",
)
ABLATION_INDEX = (0, 1, 2, 7, 8, 9)
MODES = ("center_push", "offset_push", "free_motion")
COHORTS = ("overall", "moving", "static", "contact", "no_contact") + MODES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features_from_predictions(action, out):
    """ONLY normalized current action and frozen, present-time outputs."""
    a = np.asarray(action, dtype=np.float64)
    xy = np.asarray(out["current_xy"], dtype=np.float64)
    pusher = np.asarray(out["pusher_xy"], dtype=np.float64)
    yaw = np.asarray(out["current_yaw_cs"], dtype=np.float64)
    delta = np.asarray(out["delta_xy"], dtype=np.float64)
    contact_score = np.asarray(out["contact_score_m"], dtype=np.float64)
    contact = np.asarray(out["next_contact"])
    n = a.shape[0]
    if (a.shape != (n, 2) or xy.shape != (n, 2) or
            pusher.shape != (n, 2) or yaw.shape != (n, 2) or
            delta.shape != (n, 2) or contact_score.shape != (n,) or
            contact.shape != (n,) or contact.dtype != np.bool_):
        raise ValueError("Invalid IC-01 input/output shapes or contact dtype")
    if not all(np.isfinite(x).all() for x in (a, xy, pusher, yaw, delta, contact_score)):
        raise ValueError("Nonfinite inference feature")
    if np.any(np.linalg.norm(a, axis=1) > 1.0 + 1e-5):
        raise ValueError("Action norm exceeds one")
    x = np.column_stack((
        a[:, 0], a[:, 1], np.linalg.norm(a, axis=1),
        (pusher[:, 0] - xy[:, 0]) / 0.20,
        (pusher[:, 1] - xy[:, 1]) / 0.20,
        yaw[:, 0], yaw[:, 1],
        delta[:, 0] / SCALE_M, delta[:, 1] / SCALE_M,
        contact_score / SCALE_M,
    ))
    if x.shape != (n, len(FEATURES)):
        raise AssertionError("Feature dimension mismatch")
    return np.clip(x, -4.0, 4.0), contact


def fit_ridge(x, target, active, ridge_lambda):
    """Intercept-unpenalized ridge. Stats are fitted on selected TRAIN only."""
    if ridge_lambda <= 0 or not math.isfinite(ridge_lambda):
        raise ValueError("Invalid ridge penalty")
    select = np.asarray(active, dtype=bool)
    x = np.asarray(x, dtype=np.float64)[select]
    target = np.asarray(target, dtype=np.float64)[select]
    if x.ndim != 2 or target.shape != (len(x), 2) or len(x) < 20:
        raise ValueError("Insufficient fitting examples")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite ridge data")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ (target / SCALE_M))
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite regression coefficients")
    return {"mean": mean, "std": std, "coef": coefficients,
            "lambda": float(ridge_lambda)}


def predict_ridge(x, fitted):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fitted["mean"]):
        raise ValueError("Invalid residual feature dimension")
    design = np.column_stack((np.ones(len(x)), (x - fitted["mean"]) / fitted["std"]))
    residual = (design @ fitted["coef"]) * SCALE_M
    if residual.shape != (len(x), 2) or not np.isfinite(residual).all():
        raise ValueError("Nonfinite residual prediction")
    return residual


def bounded_contact_correction(raw_correction, contact, alpha):
    """Bound each physical correction to 4 mm; never correct predicted no-contact."""
    correction = np.asarray(raw_correction, dtype=np.float64)
    active = np.asarray(contact, dtype=bool)
    if correction.ndim != 2 or correction.shape != (len(active), 2):
        raise ValueError("Invalid correction shape")
    if not np.isfinite(correction).all() or not (0 <= alpha <= 1):
        raise ValueError("Invalid correction or mixing coefficient")
    length = np.linalg.norm(correction, axis=1)
    multiplier = np.minimum(1.0, CAP_M / np.maximum(length, 1e-15))
    return correction * multiplier[:, None] * active[:, None] * alpha


def score(predicted_delta, truth_delta, predicted_current, truth_next, moving):
    """Meters in; mean Euclidean motion and next-XY errors in cm out."""
    motion = np.linalg.norm(predicted_delta - truth_delta, axis=1) * 100.0
    next_error = np.linalg.norm(predicted_current + predicted_delta - truth_next, axis=1) * 100.0
    if len(motion) == 0 or not np.isfinite(motion).all() or not np.isfinite(next_error).all():
        raise ValueError("Invalid metrics")
    if not np.any(moving) or not np.any(~moving):
        raise ValueError("Missing moving or static TRAIN-internal cohort")
    return float(motion.mean() + 0.5 * motion[moving].mean() + 0.5 * motion[~moving].mean())


def train_holdout_filenames(episodes):
    """Mode-stratified, deterministic episode-level 30/12 TRAIN-internal split."""
    by_mode = {mode: [] for mode in MODES}
    for entry in episodes:
        if entry["mode"] not in by_mode:
            raise ValueError("Unknown episode mode")
        by_mode[entry["mode"]].append(entry["file"])
    fit, holdout = set(), set()
    for mode, names in by_mode.items():
        if len(names) != 42 or len(set(names)) != 42:
            raise ValueError(f"Unexpected TRAIN episodes in {mode}")
        ordered = sorted(names, key=lambda name: (hashlib.sha256(
            f"{SEED}|{mode}|{name}".encode("utf-8")).hexdigest(), name))
        fit.update(ordered[:30])
        holdout.update(ordered[30:])
    if len(fit) != 90 or len(holdout) != 36 or fit & holdout:
        raise AssertionError("Invalid episode-level split")
    return fit, holdout


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def collect(split_name, frozen, batch_size):
    """Explicitly refuse any split except TRAIN and VALIDATION."""
    if split_name not in ("train", "val"):
        raise ValueError("M6 is restricted to TRAIN and VALIDATION")
    import torch
    from torch.utils.data import DataLoader
    from src.data.world_model_dataset import ActionWorldWindowDataset

    dataset = ActionWorldWindowDataset(split_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    arrays = {name: [] for name in (
        "x", "contact_pred", "delta_base", "current_base", "delta_true",
        "next_true", "contact_true", "episode", "mode",
    )}
    frozen.eval()
    with torch.inference_mode():
        for batch in loader:
            frames = batch["model_input"]["frames"]
            action = batch["model_input"]["action"]
            prediction = frozen(frames, action)
            out = {k: as_numpy(v) for k, v in prediction.items()}
            feature, contact = features_from_predictions(as_numpy(action), out)
            pieces = {
                "x": feature,
                "contact_pred": contact,
                "delta_base": out["delta_xy"],
                "current_base": out["current_xy"],
                "delta_true": as_numpy(batch["labels"]["delta_xy"]).astype(np.float64),
                "next_true": as_numpy(batch["labels"]["next_xy"]).astype(np.float64),
                "contact_true": as_numpy(batch["labels"]["next_contact"]) >= 0.5,
                "episode": np.asarray(batch["metadata"]["episode"], dtype=str),
                "mode": np.asarray(batch["metadata"]["mode"], dtype=str),
            }
            for key, arr in pieces.items():
                arrays[key].append(arr)
    out = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    expected = 7938 if split_name == "train" else 1701
    if len(out["x"]) != expected or any(len(a) != expected for a in out.values()):
        raise RuntimeError("Unexpected window count")
    if not all(np.isfinite(out[key]).all() for key in (
            "x", "delta_base", "current_base", "delta_true", "next_true")):
        raise ValueError("Nonfinite collected arrays")
    return out, dataset.episodes


def group_metrics(base, delta, cohort):
    """Explicit empty cohorts are None rather than misleading zeroes."""
    mask = np.asarray(cohort, dtype=bool)
    if not mask.any():
        return None
    motion = np.linalg.norm(delta[mask] - base["delta_true"][mask], axis=1) * 100.0
    next_xy = np.linalg.norm(base["current_base"][mask] + delta[mask] - base["next_true"][mask], axis=1) * 100.0
    return {
        "n": int(mask.sum()),
        "motion_cm_mean": float(motion.mean()),
        "motion_cm_p90": float(np.percentile(motion, 90)),
        "motion_cm_max": float(motion.max()),
        "next_xy_cm_mean": float(next_xy.mean()),
        "next_xy_cm_p90": float(np.percentile(next_xy, 90)),
        "next_xy_cm_max": float(next_xy.max()),
    }


def metrics(base, delta):
    moving = np.linalg.norm(base["delta_true"], axis=1) >= MOVING_M
    masks = {
        "overall": np.ones(len(moving), dtype=bool),
        "moving": moving, "static": ~moving,
        "contact": base["contact_true"], "no_contact": ~base["contact_true"],
        **{mode: base["mode"] == mode for mode in MODES},
    }
    return {name: group_metrics(base, delta, masks[name]) for name in COHORTS}


def bootstrap_episode_ci(base, base_delta, new_delta, seed=SEED, reps=2000):
    """95% percentile CI for paired per-episode mean motion-error change."""
    old = np.linalg.norm(base_delta - base["delta_true"], axis=1) * 100
    new = np.linalg.norm(new_delta - base["delta_true"], axis=1) * 100
    episodes = sorted(set(base["episode"].tolist()))
    if len(episodes) != 27:
        raise ValueError("Expected 27 validation episodes")
    diffs = np.array([float((new - old)[base["episode"] == ep].mean()) for ep in episodes])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episodes), size=(reps, len(episodes)))
    means = diffs[draws].mean(axis=1)
    return {"definition": "candidate minus IC-01; negative favors candidate",
            "point_cm": float(diffs.mean()),
            "bootstrap_95pct_cm": [float(v) for v in np.percentile(means, [2.5, 97.5])],
            "n_episodes": len(episodes), "replicates": reps}


def model_to_json(fit):
    return {"mean": fit["mean"].tolist(), "std": fit["std"].tolist(),
            "coef": fit["coef"].tolist(), "lambda": fit["lambda"]}


def write_new_json(path, obj):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite: {path}")
    payload = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(payload)



def research_markdown(report, model_hash, report_hash):
    """Human-readable, reproducible research record from one VAL scoring."""
    result = report["metrics_cm"]
    ci = report["paired_episode_bootstrap_motion_cm"]
    rows = []
    for name in ("frozen_ic01", "train_constant_residual", "m6_full_ridge", "m6_feature_ablation"):
        m = result[name]["overall"]
        rows.append(f"| {name} | {m['motion_cm_mean']:.6f} | {m['motion_cm_p90']:.6f} | {m['next_xy_cm_mean']:.6f} |")
    rows_cohort = []
    for cohort in COHORTS:
        a = result["frozen_ic01"][cohort]
        b = result["m6_full_ridge"][cohort]
        if a is None or b is None:
            rows_cohort.append(f"| {cohort} | n/a | n/a | n/a |")
        else:
            rows_cohort.append(f"| {cohort} | {a['n']} | {a['motion_cm_mean']:.6f} | {b['motion_cm_mean']:.6f} |")
    delta = report["overall_motion_cm_change_m6_minus_ic01"]
    verdict = ("Lower validation mean motion error than frozen IC-01."
               if delta < 0 else "No reduction in validation mean motion error versus IC-01.")
    return "\n".join([
        "# M6 — TRAIN-only contact-gated XY residual experiment",
        "", "## Question and hypothesis", "",
        "Can a lightweight, capped, contact-conditioned residual fitted on TRAIN episodes",
        "reduce one-step XY motion error beyond the immutable M4 IC-01 predictor?",
        "This is an experiment, not an assumed improvement or a new held-out TEST result.",
        "", "## Protocol", "",
        f"- Source file SHA-256: `{report['source_sha256']}`",
        f"- TRAIN windows: {report['n_train_windows']}; VALIDATION windows: {report['n_val_windows']}.",
        "- TRAIN episode-level selection: 30 fit / 12 select per mode (90/36 total).",
        "- Linear ridge regression on predicted-contact windows only; no future state at inference.",
        "- TRAIN-only lambda and correction strength selection. Refit full TRAIN before VAL.",
        f"- Selected TRAIN lambda / alpha: {report['model_selection']['lambda']} / {report['model_selection']['alpha']}.",
        "- Bounded XY correction: maximum 0.004 meters per step; no correction when predicted no-contact.",
        "- Baselines: frozen IC-01 and TRAIN-mean contact residual; six-feature ablation.",
        "- No checkpoint retraining, TEST access, or validation-dependent method selection.",
        "- Moving cohort: true displacement >= 0.001 m, evaluation grouping only.",
        "- Metrics: mean and 90th-percentile Euclidean XY displacement and next-position errors (cm).",
        "", "## One-pass VALIDATION results", "",
        "| Method | Motion mean (cm) | Motion P90 (cm) | Next XY mean (cm) |",
        "|---|---:|---:|---:|", *rows,
        "", "| Cohort | Windows | IC-01 motion mean (cm) | M6 motion mean (cm) |",
        "|---|---:|---:|---:|", *rows_cohort,
        "", f"- Mean change (M6 minus IC-01): **{delta:+.6f} cm** (negative = improvement).",
        f"- Paired 27-episode bootstrap 95% CI: [{ci['bootstrap_95pct_cm'][0]:+.6f}, {ci['bootstrap_95pct_cm'][1]:+.6f}] cm (2,000 draws; descriptive).",
        f"- Predeclared conclusion: **{verdict}**",
        "- An improvement in overall mean does not imply every cohort, P90, or independent domain improved.",
        "", "## Provenance and limitations", "",
        f"- Model JSON SHA-256: `{model_hash}`",
        f"- VAL report JSON SHA-256: `{report_hash}`",
        "- Exact episode filenames and parameters are in model.json and validation_report.json.",
        "- Real-observation-conditioned one-step physical-state prediction only; no next-RGB synthesis.",
        "- Scene-specific PushT colors and fixed MuJoCo physics; no unseen-domain generalization.",
        "- TEST from M4 was already consumed and was NOT opened or rerun in M6.",
        "- VALIDATION is not an independent newly held-out TEST; do not tune based on this report.",
        "- If the method does not outperform baselines, retain the negative result without adjustment.",
        "", "## Reproduction boundary", "",
        "Script entry point: `python -B -m src.research.m6_residual_v01 --run-train-val`.",
        "The original experiment is run once; rerunning would require a separately approved protocol",
        "and a new versioned output namespace. Do not rerun M4 TEST.", "",
    ])


def self_test():
    """Synthetic only: NEVER loads datasets/checkpoints/simulator."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(144, len(FEATURES)))
    active = np.arange(144) % 3 != 0
    coeff = rng.normal(size=(len(FEATURES) + 1, 2)) * 0.001
    target = (np.column_stack((np.ones(144), x)) @ coeff) * SCALE_M
    fit = fit_ridge(x, target, active, 1.0)
    p = predict_ridge(x, fit)
    assert p.shape == (144, 2) and np.isfinite(p).all()
    c = bounded_contact_correction(np.ones((4, 2)), np.array([True, False, True, False]), 1.0)
    assert np.allclose(c[[1, 3]], 0)
    assert np.all(np.linalg.norm(c, axis=1) <= CAP_M + 1e-12)
    dummy = {
        "current_xy": np.zeros((2, 2)), "pusher_xy": np.zeros((2, 2)),
        "current_yaw_cs": np.array([[1., 0.], [0., 1.]]),
        "delta_xy": np.zeros((2, 2)), "contact_score_m": np.zeros(2),
        "next_contact": np.array([True, False]),
    }
    f, flag = features_from_predictions(np.zeros((2, 2)), dummy)
    assert f.shape == (2, len(FEATURES)) and flag.tolist() == [True, False]
    try:
        features_from_predictions(np.zeros((2, 2)), {**dummy, "next_contact": np.array([1, 0])})
        raise AssertionError("Integer contact should fail")
    except ValueError:
        pass
    try:
        bounded_contact_correction(np.ones((2, 2)), np.ones(2, dtype=bool), 1.2)
        raise AssertionError("Alpha > 1 should fail")
    except ValueError:
        pass
    episodes = [{"file": f"{mode}_{i}", "mode": mode} for mode in MODES for i in range(42)]
    left, right = train_holdout_filenames(episodes)
    assert len(left) == 90 and len(right) == 36 and not (left & right)
    assert left == train_holdout_filenames(episodes)[0]
    b = {"delta_true": np.array([[0.002, 0], [0., 0.]]),
         "current_base": np.zeros((2, 2)), "next_true": np.zeros((2, 2)),
         "contact_true": np.array([True, False]), "mode": np.array(["center_push", "free_motion"])}
    gm = metrics(b, np.zeros((2, 2)))
    assert gm["overall"]["n"] == 2 and gm["offset_push"] is None
    print("M6 SYNTHETIC-ONLY SELF-TEST: PASSED")
    print("NO TRAIN, VALIDATION, TEST, MODEL INFERENCE, OR MUJOCO OPENED")


def experiment(batch_size):
    import torch
    from src.models.integrated_pusht_m4 import IntegratedPushTWorldModel
    from src.data.world_model_dataset import MANIFEST_SHA256, SPLITS_SHA256

    if OUTPUT.exists() or (ROOT / "research/M6_RESIDUAL_EXPERIMENT.md").exists():
        raise FileExistsError("M6 output/research record already exists; no reruns")
    if batch_size < 1 or batch_size > 64:
        raise ValueError("Batch size must be between 1 and 64")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    frozen = IntegratedPushTWorldModel.restore_frozen()
    if any(p.requires_grad for p in frozen.parameters()):
        raise AssertionError("Frozen model unexpectedly trainable")
    before = {name: tensor.detach().cpu().clone() for name, tensor in frozen.state_dict().items()}
    train, episodes = collect("train", frozen, batch_size)
    train_fit_eps, train_select_eps = train_holdout_filenames(episodes)
    fit_mask = np.isin(train["episode"], list(train_fit_eps))
    select_mask = np.isin(train["episode"], list(train_select_eps))
    if fit_mask.sum() != 5670 or select_mask.sum() != 2268:
        raise AssertionError("Unexpected internal training-window counts")
    target = train["delta_true"] - train["delta_base"]
    active = train["contact_pred"]
    if int((fit_mask & active).sum()) < 100:
        raise ValueError("Too few contact-positive training windows")
    moving_select = np.linalg.norm(train["delta_true"][select_mask], axis=1) >= MOVING_M
    baseline_selection = score(train["delta_base"][select_mask], train["delta_true"][select_mask],
                               train["current_base"][select_mask], train["next_true"][select_mask], moving_select)
    choices = [{"lambda": None, "alpha": 0.0, "score": baseline_selection}]
    for lam in LAMBDAS:
        fitted = fit_ridge(train["x"][fit_mask], target[fit_mask], active[fit_mask], lam)
        raw = predict_ridge(train["x"][select_mask], fitted)
        for alpha in ALPHAS:
            correction = bounded_contact_correction(raw, active[select_mask], alpha)
            candidate_delta = train["delta_base"][select_mask] + correction
            objective = score(candidate_delta, train["delta_true"][select_mask],
                              train["current_base"][select_mask], train["next_true"][select_mask], moving_select)
            choices.append({"lambda": lam, "alpha": alpha, "score": objective})
    # Predeclared deterministic tie rule: least intervention first.
    selected = min(choices, key=lambda c: (c["score"], c["alpha"], float(c["lambda"] or 0)))
    trained = fit_ridge(train["x"], target, active, selected["lambda"] if selected["lambda"] else 10.0)
    ablation = fit_ridge(train["x"][:, ABLATION_INDEX], target, active,
                         selected["lambda"] if selected["lambda"] else 10.0)
    constant = target[active].mean(axis=0)
    for name, prev in before.items():
        if not torch.equal(prev, frozen.state_dict()[name]):
            raise AssertionError(f"Frozen checkpoint state changed: {name}")

    model_record = {
        "schema": "actionworld-m6-contact-residual-v01",
        "method": "contact-gated ridge on frozen IC-01 outputs",
        "hypothesis": "bounded train-only XY residual can lower validation motion error",
        "validation_not_used_for_selection": True,
        "features": list(FEATURES), "ablation_indices": list(ABLATION_INDEX),
        "seed": SEED, "fit_only_predicted_contact": True,
        "correction_cap_m": CAP_M, "movement_threshold_m": MOVING_M,
        "choice": selected, "internal_choices": choices,
        "train_internal_fit_episodes": sorted(train_fit_eps),
        "train_internal_selection_episodes": sorted(train_select_eps),
        "full_train_active_windows": int(active.sum()),
        "trained_full": model_to_json(trained),
        "trained_ablation": model_to_json(ablation),
        "constant_residual_m": constant.tolist(),
        "source_sha256": sha256(Path(__file__)),
        "manifest_sha256": MANIFEST_SHA256,
        "splits_sha256": SPLITS_SHA256,
        "checkpoints_sha256": {
            "dynamics": sha256(ROOT / "data/processed/m4_visual_v03/best_model.pt"),
            "yaw": sha256(ROOT / "data/processed/m4_yaw_v04/best_model.pt"),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_new_json(OUTPUT / "model.json", model_record)
    print("M6 TRAIN FIT SEALED; VALIDATION NOT OPENED BEFORE TRAIN MODEL SAVED", flush=True)
    print("TRAIN selection:", json.dumps(selected, sort_keys=True), flush=True)

    val, val_episodes = collect("val", frozen, batch_size)
    if len(val_episodes) != 27:
        raise AssertionError("Unexpected validation episode count")
    for name, prev in before.items():
        if not torch.equal(prev, frozen.state_dict()[name]):
            raise AssertionError(f"Frozen checkpoint state changed: {name}")
    alpha = selected["alpha"]
    correction_full = bounded_contact_correction(predict_ridge(val["x"], trained), val["contact_pred"], alpha)
    correction_ablation = bounded_contact_correction(
        predict_ridge(val["x"][:, ABLATION_INDEX], ablation), val["contact_pred"], alpha)
    correction_constant = bounded_contact_correction(
        np.repeat(constant[None, :], len(val["x"]), axis=0), val["contact_pred"], 1.0)
    variants = {
        "frozen_ic01": val["delta_base"],
        "train_constant_residual": val["delta_base"] + correction_constant,
        "m6_full_ridge": val["delta_base"] + correction_full,
        "m6_feature_ablation": val["delta_base"] + correction_ablation,
    }
    results = {name: metrics(val, delta) for name, delta in variants.items()}
    delta_metric = (results["m6_full_ridge"]["overall"]["motion_cm_mean"] -
                    results["frozen_ic01"]["overall"]["motion_cm_mean"])
    report = {
        "schema": "actionworld-m6-val-report-v01",
        "source_sha256": sha256(Path(__file__)),
        "evaluation_split": "val", "test_access": "NEVER",
        "n_train_windows": len(train["x"]), "n_val_windows": len(val["x"]),
        "val_episodes": sorted({str(x) for x in val["episode"]}),
        "seed": SEED, "batch_size": batch_size, "numpy": np.__version__,
        "torch": torch.__version__, "python": platform.python_version(),
        "train_model_file_sha256": sha256(OUTPUT / "model.json"),
        "model_selection": selected, "metrics_cm": results,
        "paired_episode_bootstrap_motion_cm": bootstrap_episode_ci(
            val, variants["frozen_ic01"], variants["m6_full_ridge"]),
        "overall_motion_cm_change_m6_minus_ic01": float(delta_metric),
        "improves_validation_overall_motion": bool(delta_metric < 0),
        "compared_with_train_constant_residual": bool(
            results["m6_full_ridge"]["overall"]["motion_cm_mean"] <
            results["train_constant_residual"]["overall"]["motion_cm_mean"]),
        "limitations": [
            "Only single-step physical-state predictions on the same PushT environment family.",
            "IC-01 and its M4 TEST numbers remain untouched.",
            "VAL was evaluation-only; findings do not constitute a new independent TEST result.",
            "Contact-negative predictions receive no residual correction.",
            "A positive numerical change denotes worse error; no automatic claim of improvement.",
        ],
    }
    write_new_json(OUTPUT / "validation_report.json", report)
    documentation = research_markdown(
        report, sha256(OUTPUT / "model.json"), sha256(OUTPUT / "validation_report.json"))
    (ROOT / "research/M6_RESIDUAL_EXPERIMENT.md").write_text(
        documentation, encoding="utf-8")
    print("RESEARCH_DOC_SHA256:", sha256(ROOT / "research/M6_RESIDUAL_EXPERIMENT.md"))
    print(json.dumps({
        "n_train": len(train["x"]), "n_val": len(val["x"]),
        "selection": selected,
        "frozen_motion_cm": results["frozen_ic01"]["overall"]["motion_cm_mean"],
        "m6_motion_cm": results["m6_full_ridge"]["overall"]["motion_cm_mean"],
        "constant_motion_cm": results["train_constant_residual"]["overall"]["motion_cm_mean"],
        "improved": report["improves_validation_overall_motion"],
        "bootstrap_ci": report["paired_episode_bootstrap_motion_cm"],
        "model_sha256": sha256(OUTPUT / "model.json"),
        "report_sha256": sha256(OUTPUT / "validation_report.json"),
    }, indent=2, sort_keys=True, allow_nan=False))
    print("M6 TRAIN-ONLY FIT + SINGLE VALIDATION EVALUATION COMPLETE; TEST UNOPENED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run-train-val", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        experiment(args.batch_size)


if __name__ == "__main__":
    main()
