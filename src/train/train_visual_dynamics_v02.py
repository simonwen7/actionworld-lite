"""V0.2 visual physics training: pose warmup, then action-conditioned learning."""

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
from src.models.visual_dynamics import VisualPhysicalWorldModel
from src.train.train_visual_dynamics import (
    add_metric,
    contact_flags,
    early_stopping_update,
    finalize,
    move_batch,
    new_bucket,
    save_new_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/m4_visual_v02"

SOURCE_HASHES = {
    "src/data/world_model_dataset.py": "0c30473110082e2f0bc851b8f03f9a4f7c9f13dec917cfb3606022aebce96ab3",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
    "src/train/train_visual_dynamics.py": "b58eebfa61764dd98b6d6a743beadd6a2ae9ea6244e6dedfcc16620213d38edc",
}

DATA_HASHES = {
    "data/raw/m3_main_v01/manifest.json": "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50",
    "data/processed/m3_main_v01/splits.json": "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa",
}

MODES = ("center_push", "offset_push", "free_motion")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs():
    for name, expected in {**SOURCE_HASHES, **DATA_HASHES}.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source/data hash mismatch: {name}")

    for name in ("best_model.pt", "training_history.json"):
        if (OUT / name).exists():
            raise FileExistsError(f"Existing V0.2 artifact: {name}")


def training_loss_parts(pred, labels, pos_weight):
    contact = labels["next_contact"]
    motion_weights = 1.0 + contact

    def scaled_huber(a, b, scale):
        return F.smooth_l1_loss(
            a / scale, b / scale, beta=0.1, reduction="none"
        )

    def motion_mean(values):
        return (values * motion_weights).sum() / motion_weights.sum()

    current_yaw_cost = 1.0 - (
        pred["current_yaw_cs"] * labels["current_yaw_cs"]
    ).sum(dim=1)

    next_yaw_cost = 1.0 - (
        pred["next_yaw_cs"] * labels["next_yaw_cs"]
    ).sum(dim=1)

    return {
        "current_position": scaled_huber(
            pred["current_xy"], labels["current_xy"], 0.2
        ).mean(),
        "current_yaw": current_yaw_cost.mean(),
        "next_position": scaled_huber(
            pred["next_xy"], labels["next_xy"], 0.2
        ).mean(),
        "next_yaw": next_yaw_cost.mean(),
        "motion_position": motion_mean(
            scaled_huber(
                pred["delta_xy"], labels["delta_xy"], 0.02
            ).mean(dim=1)
        ),
        "motion_yaw": motion_mean(
            scaled_huber(
                pred["delta_yaw"], labels["delta_yaw"], 0.1
            )
        ),
        "contact": F.binary_cross_entropy_with_logits(
            pred["next_contact_logits"],
            contact,
            pos_weight=pos_weight,
        ),
    }


def training_objective(parts, stage):
    if stage == "pose":
        return (
            0.7 * parts["current_position"]
            + 0.7 * parts["current_yaw"]
        )

    if stage != "joint":
        raise ValueError(f"Unknown stage: {stage}")

    return (
        0.7 * parts["current_position"]
        + 0.5 * parts["next_position"]
        + 1.5 * parts["motion_position"]
        + 0.4 * parts["current_yaw"]
        + 0.3 * parts["next_yaw"]
        + 0.6 * parts["motion_yaw"]
        + 0.35 * parts["contact"]
    )


def train_epoch(
    model, loader, optimizer, device, positive_weight,
    stage, limit=None
):
    model.train()
    loss_sum = 0.0
    count = 0

    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break

        inputs, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        prediction = model(
            inputs["frames"], inputs["action"]
        )

        loss = training_objective(
            training_loss_parts(
                prediction, labels, positive_weight
            ),
            stage,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite training objective")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        n = inputs["frames"].shape[0]
        loss_sum += loss.detach().item() * n
        count += n

    if count == 0:
        raise RuntimeError("No training batches")

    return loss_sum / count, count


@torch.no_grad()
def evaluate(
    model, loader, device, positive_weight,
    stage, limit=None
):
    model.eval()

    buckets = {
        key: new_bucket()
        for key in (
            "overall", "contact", "no_contact", *MODES
        )
    }

    loss_sum = 0.0
    count = 0

    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break

        inputs, labels = move_batch(batch, device)

        prediction = model(
            inputs["frames"], inputs["action"]
        )

        loss = training_objective(
            training_loss_parts(
                prediction, labels, positive_weight
            ),
            stage,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite validation objective")

        n = inputs["frames"].shape[0]
        loss_sum += loss.detach().item() * n
        count += n

        pos = torch.linalg.vector_norm(
            prediction["next_xy"] - labels["next_xy"],
            dim=1,
        ) * 100.0

        dot = (
            prediction["next_yaw_cs"]
            * labels["next_yaw_cs"]
        ).sum(dim=1).clamp(-1, 1)

        cross = (
            prediction["next_yaw_cs"][:, 1]
            * labels["next_yaw_cs"][:, 0]
            - prediction["next_yaw_cs"][:, 0]
            * labels["next_yaw_cs"][:, 1]
        )

        yaw = torch.atan2(
            cross.abs(), dot
        ) * (180.0 / math.pi)

        predicted_contact = (
            prediction["next_contact_logits"] >= 0
        )

        actual_contact = (
            labels["next_contact"] >= 0.5
        )

        for j, mode in enumerate(
            batch["metadata"]["mode"]
        ):
            is_contact = bool(actual_contact[j].item())
            estimated = bool(predicted_contact[j].item())

            group = (
                "contact" if is_contact else "no_contact"
            )

            for name in ("overall", mode, group):
                add_metric(
                    buckets[name],
                    pos[j].item(),
                    yaw[j].item(),
                    estimated,
                    is_contact,
                )

    if count == 0:
        raise RuntimeError("No validation batches")

    return (
        loss_sum / count,
        {
            name: finalize(bucket)
            for name, bucket in buckets.items()
        },
    )


def clone_weights(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device", choices=("cpu", "mps"), default="cpu"
    )
    parser.add_argument(
        "--pose-epochs", type=int, default=8
    )
    parser.add_argument(
        "--pose-patience", type=int, default=3
    )
    parser.add_argument(
        "--epochs", type=int, default=20
    )
    parser.add_argument(
        "--patience", type=int, default=4
    )
    parser.add_argument(
        "--batch-size", type=int, default=16
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4
    )
    parser.add_argument(
        "--seed", type=int, default=20261025
    )

    args = parser.parse_args()

    if (
        args.pose_epochs < 1
        or args.pose_patience < 1
        or args.epochs < 1
        or args.patience < 1
        or args.batch_size < 2
        or not (0 < args.lr < 1)
    ):
        parser.error("Invalid training arguments")

    if args.smoke and args.device != "cpu":
        parser.error("Smoke is CPU only")

    if (
        args.device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS unavailable; no automatic CPU fallback"
        )

    verify_inputs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.smoke:
        torch.set_num_threads(2)

    train = ActionWorldWindowDataset("train")
    val = ActionWorldWindowDataset("val")

    flags = contact_flags(train)

    positives = int(flags.sum())
    negatives = len(flags) - positives

    assert len(train) == 7938
    assert len(val) == 1701
    assert positives == 2244
    assert negatives == 5694

    ratio = negatives / positives
    positive_weight_value = math.sqrt(ratio)

    natural_rate = positives / len(flags)

    constant_optimum = (
        natural_rate * positive_weight_value
        / (
            natural_rate * positive_weight_value
            + 1 - natural_rate
        )
    )

    assert constant_optimum < 0.5

    print(
        f"TRAIN={len(train)} VAL={len(val)} "
        f"CONTACT={positives} "
        f"POS_WEIGHT={positive_weight_value:.6f} "
        f"CONSTANT_CONTACT_OPTIMUM={constant_optimum:.6f} "
        f"TEST=UNOPENED",
        flush=True,
    )

    generator = torch.Generator().manual_seed(
        args.seed
    )

    # Natural-frequency sampling, not class oversampling.
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)

    pos_weight = torch.tensor(
        positive_weight_value,
        device=device,
        dtype=torch.float32,
    )

    model = VisualPhysicalWorldModel().to(device)

    original_encoder = (
        model.encoder[0].weight.detach().cpu().clone()
    )

    original_dynamics = (
        model.action_dynamics_head[0]
        .weight.detach().cpu().clone()
    )

    stage_config = (
        (
            "pose",
            args.pose_epochs,
            args.pose_patience,
        ),
        (
            "joint",
            args.epochs,
            args.patience,
        ),
    )

    histories = {}
    stage_best_epoch = {}
    stage_best_loss = {}

    limit = 2 if args.smoke else None

    for stage, max_epochs, patience in stage_config:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )

        best_loss = float("inf")
        best_epoch = None
        best_weights = None
        stale = 0
        records = []

        n_epochs = 1 if args.smoke else max_epochs

        for epoch in range(1, n_epochs + 1):
            train_loss, train_n = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight,
                stage,
                limit,
            )

            val_loss, metrics = evaluate(
                model,
                val_loader,
                device,
                pos_weight,
                stage,
                limit,
            )

            records.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": metrics,
            })

            print(
                f"STAGE {stage} EPOCH {epoch:03d}: "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"train_n={train_n} "
                f"val_n={metrics['overall']['n']}",
                flush=True,
            )

            (
                best_loss,
                stale,
                improved,
                should_stop,
            ) = early_stopping_update(
                val_loss,
                best_loss,
                stale,
                patience,
            )

            if improved:
                best_epoch = epoch
                best_weights = clone_weights(model)

            if should_stop and not args.smoke:
                print(
                    f"STAGE {stage} EARLY STOP: {epoch} "
                    f"BEST: {best_epoch}",
                    flush=True,
                )
                break

        assert best_weights is not None
        assert best_epoch is not None

        model.load_state_dict(
            best_weights,
            strict=True,
        )

        histories[stage] = records
        stage_best_epoch[stage] = best_epoch
        stage_best_loss[stage] = best_loss

        if args.smoke:
            assert (
                records[0]["val_metrics"]["overall"]["n"]
                == 2 * args.batch_size
            )

    if args.smoke:
        assert not torch.equal(
            original_encoder,
            model.encoder[0].weight.detach().cpu(),
        )

        assert not torch.equal(
            original_dynamics,
            model.action_dynamics_head[0]
            .weight.detach().cpu(),
        )

        print(
            "PASS: pose and joint CPU optimizer updates"
        )
        print(
            "PASS: smoke created no model/history artifacts"
        )
        print("M4 V02 TRAINING CPU SMOKE: PASSED")
        return

    config = {
        "schema_version": "m4-visual-physics-v02",
        "device": args.device,
        "pose_epochs_requested": args.pose_epochs,
        "pose_patience": args.pose_patience,
        "joint_epochs_requested": args.epochs,
        "joint_patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "sampling": (
            "natural_frequency_shuffle_no_replacement"
        ),
        "contact_pos_weight": positive_weight_value,
        "action_input": (
            "two_RGB_frames_and_current_action_only"
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)

    checkpoint_file = OUT / "best_model.pt"
    history_file = OUT / "training_history.json"

    source_hashes = dict(SOURCE_HASHES)
    source_hashes[
        "src/train/train_visual_dynamics_v02.py"
    ] = sha256(Path(__file__))

    payload = {
        "model": "VisualPhysicalWorldModel",
        "parameter_count": model.parameter_count(),
        "state_dict": clone_weights(model),
        "config": config,
        "best_stage": "joint",
        "best_epoch": stage_best_epoch["joint"],
        "best_val_loss": stage_best_loss["joint"],
        "source_sha256": source_hashes,
    }

    save_new_atomic(
        checkpoint_file,
        lambda path: torch.save(payload, path),
    )

    report = {
        "schema_version": "m4-visual-training-v02",
        "config": config,
        "stage_history": histories,
        "stage_best_epoch": stage_best_epoch,
        "stage_best_loss": stage_best_loss,
        "checkpoint_sha256": sha256(checkpoint_file),
        "source_sha256": source_hashes,
        "data_sha256": DATA_HASHES,
        "evaluation_policy": (
            "Training and validation only; test unopened"
        ),
    }

    save_new_atomic(
        history_file,
        lambda path: path.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        ),
    )

    print(
        "CHECKPOINT:",
        checkpoint_file,
        flush=True,
    )
    print(
        "CHECKPOINT SHA256:",
        sha256(checkpoint_file),
        flush=True,
    )
    print(
        "HISTORY:",
        history_file,
        flush=True,
    )
    print("M4 V02 FULL TRAINING: PASSED", flush=True)


if __name__ == "__main__":
    main()
