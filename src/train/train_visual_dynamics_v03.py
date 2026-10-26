"""M4 V0.3: frozen visual pose, supervised motion gate and dynamics refinement.

Train/validation labels are used for supervision and evaluation only.
The model forward pass accepts RGB frames and the current action only.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.world_model_dataset import ActionWorldWindowDataset
from src.models.visual_dynamics_v03 import GatedVisualPhysicalWorldModel
from src.train.train_visual_dynamics import early_stopping_update, save_new_atomic

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/m4_visual_v03"
V02_CHECKPOINT = ROOT / "data/processed/m4_visual_v02/best_model.pt"

FROZEN_HASHES = {
    "src/data/world_model_dataset.py": "0c30473110082e2f0bc851b8f03f9a4f7c9f13dec917cfb3606022aebce96ab3",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
    "src/models/visual_dynamics_v03.py": "d50c8debb7b04477861d248e4c6e5ec87ca63b247f48231bb4a9aa198619a659",
    "src/train/train_visual_dynamics.py": "b58eebfa61764dd98b6d6a743beadd6a2ae9ea6244e6dedfcc16620213d38edc",
    "src/train/train_visual_dynamics_v02.py": "e2c0f003ad956c39ba198f11b6fc5f5e06cf038e9e5a662d15a4eacd9e872963",
    "data/raw/m3_main_v01/manifest.json": "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50",
    "data/processed/m3_main_v01/splits.json": "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa",
    "data/processed/m4_visual_v01/best_model.pt": "3af77d096be257f3b0f806474a3995ead252f4bc0fb25e4543f248dfd59926d5",
    "data/processed/m4_visual_v01/training_history.json": "234bf0a9b24acb15e44debb63a8a39c6abd1e7b3cf93bc1c90681c79256bdaf4",
    "data/processed/m4_visual_v02/best_model.pt": "2b0417fdefcc188b4852bbc7c472c3cf3136ddde00b70e621c3c074db2d65463",
    "data/processed/m4_visual_v02/training_history.json": "546184c1d0824075a7123c5f6b7993594afc90e7afc5c507e57f0932a1890ba2",
}

MODES = ("center_push", "offset_push", "free_motion")
GROUPS = ("overall", "moving", "static", "contact", "no_contact", *MODES)
MOVING_THRESHOLD_M = 0.001  # 0.1 cm; supervision/evaluation only
MOTION_FIELDS = ("motion_cm", "motion_yaw_deg", "next_position_cm", "next_yaw_deg", "activity_prob")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs():
    for name, expected in FROZEN_HASHES.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("Frozen file changed: " + name)
    for filename in ("best_model.pt", "training_history.json"):
        if (OUT / filename).exists():
            raise FileExistsError("Refusing to overwrite V0.3 artifact: " + filename)


def new_model_from_v02():
    # The checkpoint has already been hash-verified by verify_frozen_inputs.
    checkpoint = torch.load(V02_CHECKPOINT, map_location="cpu", weights_only=True)
    assert checkpoint["parameter_count"] == 415896
    assert checkpoint["model"] == "VisualPhysicalWorldModel"
    model = GatedVisualPhysicalWorldModel()
    status = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert set(status.missing_keys) == {
        "motion_gate_head.0.weight", "motion_gate_head.0.bias",
        "motion_gate_head.2.weight", "motion_gate_head.2.bias",
    }
    assert status.unexpected_keys == []
    assert model.parameter_count() == 424345
    return model


def configure_stage(model, stage):
    # Both stages keep the pretrained visual encoder and pose head frozen.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.motion_gate_head.parameters():
        parameter.requires_grad_(True)
    if stage == "joint":
        for parameter in model.action_dynamics_head.parameters():
            parameter.requires_grad_(True)
    elif stage != "gate":
        raise ValueError("Unknown training stage: " + stage)
    return [p for p in model.parameters() if p.requires_grad]


def activity_target(labels):
    return (torch.linalg.vector_norm(labels["delta_xy"], dim=1)
            >= MOVING_THRESHOLD_M).to(dtype=torch.float32)


def loss_parts(output, labels):
    moving = activity_target(labels)
    sample_weight = 1.0 + 2.0 * moving
    pos_error = F.smooth_l1_loss(
        output["delta_xy"] / 0.02,
        labels["delta_xy"] / 0.02,
        reduction="none", beta=0.1,
    ).mean(dim=1)
    yaw_error = F.smooth_l1_loss(
        output["delta_yaw"] / 0.1,
        labels["delta_yaw"] / 0.1,
        reduction="none", beta=0.1,
    )
    magnitude_error = F.smooth_l1_loss(
        torch.linalg.vector_norm(output["delta_xy"], dim=1) / 0.02,
        torch.linalg.vector_norm(labels["delta_xy"], dim=1) / 0.02,
        reduction="none", beta=0.1,
    )
    return {
        "gate": F.binary_cross_entropy_with_logits(output["motion_logits"], moving),
        "motion_xy": (pos_error * sample_weight).sum() / sample_weight.sum(),
        "motion_yaw": (yaw_error * sample_weight).sum() / sample_weight.sum(),
        "moving_magnitude": (magnitude_error * moving).sum() / moving.sum().clamp(min=1),
        "contact_retention": F.binary_cross_entropy_with_logits(
            output["next_contact_logits"], labels["next_contact"]
        ),
    }


def objective(parts, stage):
    if stage == "gate":
        return parts["gate"]
    if stage == "joint":
        return (1.4 * parts["motion_xy"]
                + 0.4 * parts["motion_yaw"]
                + 0.3 * parts["moving_magnitude"]
                + 0.6 * parts["gate"]
                + 0.1 * parts["contact_retention"])
    raise ValueError("Unknown training stage: " + stage)


def move_batch(batch, device):
    return ({key: value.to(device) for key, value in batch["model_input"].items()},
            {key: value.to(device) for key, value in batch["labels"].items()})


def train_epoch(model, loader, optimizer, device, stage, limit=None):
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        inputs, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs["frames"], inputs["action"])
        loss = objective(loss_parts(output, labels), stage)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        n = inputs["frames"].shape[0]
        loss_sum += float(loss.detach().cpu()) * n
        sample_count += n
    if not sample_count:
        raise RuntimeError("Training produced no batches")
    return loss_sum / sample_count, sample_count


def new_bucket():
    return {"n": 0, "sums": {field: 0.0 for field in MOTION_FIELDS},
            "tp": 0, "fp": 0, "tn": 0, "fn": 0}


def accumulate(bucket, values, mask, moving, activity_pred):
    n = int(mask.sum())
    bucket["n"] += n
    if not n:
        return
    for name in MOTION_FIELDS:
        bucket["sums"][name] += float(values[name][mask].sum(dtype=np.float64))
    target = moving[mask]
    predicted = activity_pred[mask]
    bucket["tp"] += int(np.logical_and(predicted, target).sum())
    bucket["fp"] += int(np.logical_and(predicted, ~target).sum())
    bucket["tn"] += int(np.logical_and(~predicted, ~target).sum())
    bucket["fn"] += int(np.logical_and(~predicted, target).sum())


def finish(bucket):
    n = bucket["n"]
    if n == 0:
        return None
    tp, fp, tn, fn = (bucket[k] for k in ("tp", "fp", "tn", "fn"))
    return {
        "n": n,
        **{key: bucket["sums"][key] / n for key in MOTION_FIELDS},
        "activity_tp": tp, "activity_fp": fp, "activity_tn": tn, "activity_fn": fn,
        "activity_precision": tp / (tp + fp) if tp + fp else None,
        "activity_recall": tp / (tp + fn) if tp + fn else None,
        "activity_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


@torch.no_grad()
def evaluate(model, loader, device, stage, limit=None):
    model.eval()
    buckets = {group: new_bucket() for group in GROUPS}
    objective_sum = 0.0
    sample_count = 0
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        inputs, labels = move_batch(batch, device)
        output = model(inputs["frames"], inputs["action"])
        part = loss_parts(output, labels)
        loss = objective(part, stage)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite validation loss")
        n = inputs["frames"].shape[0]
        objective_sum += float(loss.detach().cpu()) * n
        sample_count += n
        # Copy whole batches to CPU once: avoid per-example Apple MPS synchronization.
        def arr(tensor):
            return tensor.detach().cpu().numpy()
        true_delta, pred_delta = arr(labels["delta_xy"]), arr(output["delta_xy"])
        true_yaw, pred_yaw = arr(labels["delta_yaw"]), arr(output["delta_yaw"])
        true_next_xy, pred_next_xy = arr(labels["next_xy"]), arr(output["next_xy"])
        true_cs, pred_cs = arr(labels["next_yaw_cs"]), arr(output["next_yaw_cs"])
        moving = np.linalg.norm(true_delta, axis=1) >= MOVING_THRESHOLD_M
        activity_prob = arr(output["motion_probability"])
        activity_pred = activity_prob >= 0.5
        true_contact = arr(labels["next_contact"]) >= 0.5
        mode = np.asarray(batch["metadata"]["mode"])
        true_angle = np.arctan2(true_cs[:, 1], true_cs[:, 0])
        pred_angle = np.arctan2(pred_cs[:, 1], pred_cs[:, 0])
        yaw_diff = np.arctan2(np.sin(pred_angle-true_angle), np.cos(pred_angle-true_angle))
        motion_yaw_diff = np.arctan2(np.sin(pred_yaw-true_yaw), np.cos(pred_yaw-true_yaw))
        values = {
            "motion_cm": np.linalg.norm(pred_delta-true_delta, axis=1)*100.0,
            "motion_yaw_deg": np.abs(motion_yaw_diff)*(180.0/math.pi),
            "next_position_cm": np.linalg.norm(pred_next_xy-true_next_xy, axis=1)*100.0,
            "next_yaw_deg": np.abs(yaw_diff)*(180.0/math.pi),
            "activity_prob": activity_prob,
        }
        for value in values.values():
            if not np.isfinite(value).all():
                raise RuntimeError("Nonfinite evaluation metric")
        masks = {
            "overall": np.ones(n, dtype=bool),
            "moving": moving,
            "static": ~moving,
            "contact": true_contact,
            "no_contact": ~true_contact,
            **{name: mode == name for name in MODES},
        }
        for name, mask in masks.items():
            accumulate(buckets[name], values, mask, moving, activity_pred)
    if not sample_count:
        raise RuntimeError("Validation produced no batches")
    metrics = {key: finish(bucket) for key, bucket in buckets.items()}
    overall = metrics["overall"]
    assert overall["n"] == sample_count
    if stage == "gate":
        selection = objective_sum / sample_count  # Natural-frequency activity BCE.
    elif stage == "joint":
        if metrics["moving"] is None or metrics["static"] is None:
            # A two-batch CPU smoke may not contain both cohorts.
            # Full validation MUST contain both; never use this fallback there.
            if limit is None:
                raise RuntimeError("Full validation missing a motion cohort")
            selection = overall["motion_cm"]
        else:
            selection = (0.50 * metrics["moving"]["motion_cm"]
                         + 0.25 * metrics["static"]["motion_cm"]
                         + 0.25 * overall["motion_cm"])
    else:
        raise ValueError("Unknown evaluation stage: " + stage)
    if not math.isfinite(selection):
        raise RuntimeError("Nonfinite validation selection score")
    return objective_sum / sample_count, selection, metrics


def clone_weights(model):
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--gate-epochs", type=int, default=6)
    parser.add_argument("--gate-patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if (min(args.gate_epochs, args.gate_patience, args.epochs,
            args.patience) < 1 or args.batch_size < 2 or not (0 < args.lr < 1)):
        parser.error("Invalid training parameters")
    if args.smoke and args.device != "cpu":
        parser.error("Smoke test must run on CPU")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    verify_frozen_inputs()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.smoke:
        torch.set_num_threads(2)
    train = ActionWorldWindowDataset("train")
    val = ActionWorldWindowDataset("val")
    assert len(train) == 7938 and len(val) == 1701
    train_activity_count = sum(
        int(np.linalg.norm(train.episodes[episode]["states"][t+1, 3:5]
                           - train.episodes[episode]["states"][t, 3:5])
            >= MOVING_THRESHOLD_M)
        for episode, t in train.windows
    )
    assert train_activity_count == 2143
    print(f"TRAIN={len(train)} TRAIN_MOVING={train_activity_count} "
          f"TRAIN_STATIC={len(train)-train_activity_count} "
          f"VAL={len(val)} TEST=UNOPENED", flush=True)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
    )
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = new_model_from_v02().to(device)
    base_encoder = clone_weights(model)["encoder.0.weight"]
    base_pose = clone_weights(model)["current_pose_head.0.weight"]
    initial_gate = clone_weights(model)["motion_gate_head.0.weight"]
    initial_dynamics = clone_weights(model)["action_dynamics_head.0.weight"]
    histories = {}
    best_epochs = {}
    best_scores = {}
    trainable_counts = {}
    for stage, max_epochs, patience in (
        ("gate", args.gate_epochs, args.gate_patience),
        ("joint", args.epochs, args.patience),
    ):
        params = configure_stage(model, stage)
        trainable_counts[stage] = sum(p.numel() for p in params)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        best_score, best_weights, best_epoch, stale = float("inf"), None, None, 0
        records = []
        for epoch in range(1, 2 if args.smoke else max_epochs+1):
            train_loss, train_n = train_epoch(
                model, train_loader, optimizer, device, stage,
                limit=2 if args.smoke else None,
            )
            val_loss, score, metrics = evaluate(
                model, val_loader, device, stage,
                limit=2 if args.smoke else None,
            )
            records.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "selection_score": score,
                            "val_metrics": metrics})
            print(f"STAGE {stage} EPOCH {epoch:03d}: "
                  f"train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} score={score:.6f} "
                  f"train_n={train_n} val_n={metrics['overall']['n']}", flush=True)
            best_score, stale, improved, stop = early_stopping_update(
                score, best_score, stale, patience,
            )
            if improved:
                best_epoch, best_weights = epoch, clone_weights(model)
            if stop and not args.smoke:
                print(f"STAGE {stage} EARLY STOP: epoch={epoch} "
                      f"best_epoch={best_epoch}", flush=True)
                break
        assert best_weights is not None and best_epoch is not None
        model.load_state_dict(best_weights, strict=True)
        histories[stage] = records
        best_epochs[stage] = best_epoch
        best_scores[stage] = best_score
    assert torch.equal(base_encoder, clone_weights(model)["encoder.0.weight"])
    assert torch.equal(base_pose, clone_weights(model)["current_pose_head.0.weight"])
    assert torch.equal(initial_gate, clone_weights(model)["motion_gate_head.0.weight"]) is False
    assert torch.equal(initial_dynamics, clone_weights(model)["action_dynamics_head.0.weight"]) is False
    if args.smoke:
        assert all(len(v) == 1 for v in histories.values())
        assert all(v[0]["val_metrics"]["overall"]["n"] ==
                   2 * args.batch_size for v in histories.values())
        print("PASS: warm-started pose and encoder frozen")
        print("PASS: gate and action dynamics optimizer updates")
        print("PASS: no training artifacts created")
        print("M4 V03 TRAINING CPU SMOKE: PASSED")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "m4-gated-motion-v03",
        "device": args.device,
        "gate_epochs_requested": args.gate_epochs,
        "gate_patience": args.gate_patience,
        "joint_epochs_requested": args.epochs,
        "joint_patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "trainable_parameters": trainable_counts,
        "pose_encoder_frozen": True,
        "motion_threshold_m": MOVING_THRESHOLD_M,
        "sampling": "natural_shuffle",
        "selection": "gate_BCE_then_0.50_moving+0.25_static+0.25_overall_motion_cm",
        "warm_start_checkpoint_sha256": FROZEN_HASHES[
            "data/processed/m4_visual_v02/best_model.pt"],
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    source_hashes = dict(FROZEN_HASHES)
    source_hashes["src/train/train_visual_dynamics_v03.py"] = sha256(Path(__file__))
    checkpoint_file = OUT / "best_model.pt"
    history_file = OUT / "training_history.json"
    checkpoint = {
        "model": "GatedVisualPhysicalWorldModel",
        "parameter_count": model.parameter_count(),
        "state_dict": clone_weights(model),
        "config": config,
        "best_epoch": best_epochs["joint"],
        "best_validation_score": best_scores["joint"],
        "source_sha256": source_hashes,
    }
    save_new_atomic(checkpoint_file, lambda temp: torch.save(checkpoint, temp))
    report = {
        "schema_version": "m4-visual-training-v03",
        "config": config,
        "stage_history": histories,
        "stage_best_epoch": best_epochs,
        "stage_best_score": best_scores,
        "checkpoint_sha256": sha256(checkpoint_file),
        "source_sha256": source_hashes,
        "evaluation_policy": "TRAIN+VALIDATION ONLY; TEST unopened",
    }
    save_new_atomic(
        history_file,
        lambda temp: temp.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n",
            encoding="utf-8",
        ),
    )
    print("CHECKPOINT:", checkpoint_file.resolve())
    print("CHECKPOINT SHA256:", sha256(checkpoint_file))
    print("HISTORY:", history_file.resolve())
    print("M4 V03 FULL TRAINING: PASSED")


if __name__ == "__main__":
    main()
