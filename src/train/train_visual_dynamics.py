"""Train the M4 visual action-conditioned one-step physical world model."""

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.world_model_dataset import ActionWorldWindowDataset
from src.models.visual_dynamics import VisualPhysicalWorldModel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/m4_visual_v01"
SOURCE_HASHES = {
    "src/data/world_model_dataset.py": "0c30473110082e2f0bc851b8f03f9a4f7c9f13dec917cfb3606022aebce96ab3",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
}
MODES = ("center_push", "offset_push", "free_motion")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources():
    for name, expected in SOURCE_HASHES.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def contact_flags(dataset):
    return np.asarray([
        int(dataset.episodes[episode_id]["states"][t + 1, 13])
        for episode_id, t in dataset.windows
    ], dtype=np.int64)


def loss_parts(prediction, labels, positive_weight):
    current_xy = labels["current_xy"]
    next_xy = labels["next_xy"]
    delta_xy = labels["delta_xy"]
    contact = labels["next_contact"]
    weights = 1.0 + 2.0 * contact

    def smooth(a, b):
        return F.smooth_l1_loss(
            a, b, reduction="none", beta=0.1
        )

    def weighted_mean(values):
        return (values * weights).sum() / weights.sum()

    current_position = smooth(
        prediction["current_xy"] / 0.2,
        current_xy / 0.2,
    ).mean()

    next_position = smooth(
        prediction["next_xy"] / 0.2,
        next_xy / 0.2,
    ).mean()

    motion_position = weighted_mean(
        smooth(
            prediction["delta_xy"] / 0.02,
            delta_xy / 0.02,
        ).mean(dim=1)
    )

    current_yaw = (
        1.0 - (
            prediction["current_yaw_cs"]
            * labels["current_yaw_cs"]
        ).sum(dim=1)
    ).mean()

    next_yaw = (
        1.0 - (
            prediction["next_yaw_cs"]
            * labels["next_yaw_cs"]
        ).sum(dim=1)
    ).mean()

    motion_yaw = weighted_mean(
        smooth(
            prediction["delta_yaw"] / 0.1,
            labels["delta_yaw"] / 0.1,
        )
    )

    contact_loss = F.binary_cross_entropy_with_logits(
        prediction["next_contact_logits"],
        contact,
        pos_weight=positive_weight,
    )

    return {
        "current_position": current_position,
        "next_position": next_position,
        "motion_position": motion_position,
        "current_yaw": current_yaw,
        "next_yaw": next_yaw,
        "motion_yaw": motion_yaw,
        "contact": contact_loss,
    }


def total_loss(parts):
    return (
        0.7 * parts["current_position"]
        + 0.7 * parts["next_position"]
        + 1.5 * parts["motion_position"]
        + 0.7 * parts["current_yaw"]
        + 0.7 * parts["next_yaw"]
        + 0.8 * parts["motion_yaw"]
        + 0.25 * parts["contact"]
    )


def move_batch(batch, device):
    x = {
        k: v.to(device)
        for k, v in batch["model_input"].items()
    }
    y = {
        k: v.to(device)
        for k, v in batch["labels"].items()
    }
    return x, y


def train_epoch(
    model, loader, optimizer, device, positive_weight, limit=None
):
    model.train()
    loss_sum = 0.0
    count = 0

    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break

        inputs, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        prediction = model(
            inputs["frames"], inputs["action"]
        )

        objective = total_loss(
            loss_parts(prediction, labels, positive_weight)
        )

        if not torch.isfinite(objective).item():
            raise RuntimeError("Nonfinite training loss")

        objective.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        n = inputs["frames"].shape[0]
        loss_sum += float(objective.detach().cpu()) * n
        count += n

    if not count:
        raise RuntimeError("No training batches")

    return loss_sum / count, count


def new_bucket():
    return {
        "n": 0,
        "pos_cm_sum": 0.0,
        "yaw_deg_sum": 0.0,
        "correct": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def add_metric(
    bucket, pos_cm, yaw_deg, predicted_contact, true_contact
):
    bucket["n"] += 1
    bucket["pos_cm_sum"] += float(pos_cm)
    bucket["yaw_deg_sum"] += float(yaw_deg)
    bucket["correct"] += int(
        predicted_contact == true_contact
    )
    bucket["tp"] += int(
        predicted_contact and true_contact
    )
    bucket["fp"] += int(
        predicted_contact and not true_contact
    )
    bucket["fn"] += int(
        not predicted_contact and true_contact
    )


def finalize(bucket):
    n = bucket["n"]

    if n == 0:
        return None

    tp = bucket["tp"]
    fp = bucket["fp"]
    fn = bucket["fn"]

    return {
        "n": n,
        "position_mean_l2_cm": (
            bucket["pos_cm_sum"] / n
        ),
        "yaw_mae_deg": (
            bucket["yaw_deg_sum"] / n
        ),
        "contact_accuracy": bucket["correct"] / n,
        "contact_precision": (
            tp / (tp + fp) if tp + fp else None
        ),
        "contact_recall": (
            tp / (tp + fn) if tp + fn else None
        ),
    }


@torch.no_grad()
def validate(
    model, loader, device, positive_weight, limit=None
):
    model.eval()

    buckets = {
        "overall": new_bucket(),
        "contact": new_bucket(),
        "no_contact": new_bucket(),
        **{m: new_bucket() for m in MODES},
    }

    loss_sum = 0.0
    count = 0

    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break

        inputs, labels = move_batch(batch, device)

        prediction = model(
            inputs["frames"], inputs["action"]
        )

        objective = total_loss(
            loss_parts(prediction, labels, positive_weight)
        )

        if not torch.isfinite(objective).item():
            raise RuntimeError("Nonfinite validation loss")

        n = inputs["frames"].shape[0]
        loss_sum += float(objective.detach().cpu()) * n
        count += n

        pos = torch.linalg.vector_norm(
            prediction["next_xy"] - labels["next_xy"],
            dim=-1,
        ) * 100

        pred_yaw = torch.atan2(
            prediction["next_yaw_cs"][:, 1],
            prediction["next_yaw_cs"][:, 0],
        )
        true_yaw = torch.atan2(
            labels["next_yaw_cs"][:, 1],
            labels["next_yaw_cs"][:, 0],
        )

        diff = pred_yaw - true_yaw

        yaw = torch.abs(
            torch.atan2(
                torch.sin(diff),
                torch.cos(diff),
            )
        ) * (180 / math.pi)

        contact_pred = (
            prediction["next_contact_logits"] >= 0
        ).cpu().numpy().astype(bool)

        contact_true = (
            labels["next_contact"]
        ).cpu().numpy().astype(bool)

        pos_np = pos.cpu().numpy()
        yaw_np = yaw.cpu().numpy()

        for j, mode in enumerate(
            batch["metadata"]["mode"]
        ):
            group = (
                "contact"
                if contact_true[j]
                else "no_contact"
            )

            for key in ("overall", mode, group):
                add_metric(
                    buckets[key],
                    pos_np[j],
                    yaw_np[j],
                    bool(contact_pred[j]),
                    bool(contact_true[j]),
                )

    if not count:
        raise RuntimeError("No validation batches")

    return (
        loss_sum / count,
        {key: finalize(b) for key, b in buckets.items()},
    )


def save_new_atomic(path, writer):
    if path.exists():
        raise FileExistsError(
            f"Refusing overwrite: {path}"
        )

    fd, name = tempfile.mkstemp(
        dir=path.parent,
        prefix="tmp_m4_",
        suffix=path.suffix,
    )
    os.close(fd)
    temp = Path(name)

    try:
        writer(temp)

        if path.exists():
            raise FileExistsError(
                f"Refusing overwrite: {path}"
            )

        os.replace(temp, path)

    finally:
        temp.unlink(missing_ok=True)


def early_stopping_update(
    val_loss, best_loss, bad_epochs, patience
):
    """Update validation best and patience counter."""
    if not math.isfinite(val_loss):
        raise ValueError("Nonfinite validation loss")
    if patience < 1 or bad_epochs < 0:
        raise ValueError("Invalid early stopping state")

    improved = val_loss < best_loss

    if improved:
        return val_loss, 0, True, False

    bad_epochs += 1
    return best_loss, bad_epochs, False, bad_epochs >= patience


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke: 2 train/val batches, no files",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261025,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 2
        or args.patience < 1
        or not (0 < args.lr < 1)
    ):
        parser.error(
            "Invalid epochs, batch size or learning rate"
        )

    if args.smoke and args.device != "cpu":
        parser.error(
            "Smoke tests must use CPU"
        )

    if (
        not args.smoke
        and args.device == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS unavailable; no automatic CPU fallback"
        )

    if (
        not args.smoke
        and (OUT / "best_model.pt").exists()
    ):
        raise FileExistsError(
            "Existing best_model.pt; refusing to overwrite"
        )

    if (
        not args.smoke
        and (OUT / "training_history.json").exists()
    ):
        raise FileExistsError(
            "Existing training_history.json; refusing to overwrite"
        )

    check_sources()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.smoke:
        torch.set_num_threads(2)

    train_data = ActionWorldWindowDataset("train")
    val_data = ActionWorldWindowDataset("val")

    train_flags = contact_flags(train_data)

    assert len(train_flags) == 7938
    assert 0 < train_flags.sum() < len(train_flags)
    assert len(val_data) == 1701

    balance = float(
        (len(train_flags) - train_flags.sum())
        / train_flags.sum()
    )

    print(
        f"TRAIN: {len(train_data)} windows;"
        f" positive contact: {int(train_flags.sum())};"
        f" pos_weight: {balance:.6f}"
    )

    print(
        f"VAL: {len(val_data)} windows;"
        " TEST never opened"
    )

    sampler_weights = torch.tensor(
        1.0 + 2.0 * train_flags,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        sampler_weights,
        num_samples=len(train_data),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device)
    model = VisualPhysicalWorldModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    positive_weight = torch.tensor(
        balance,
        dtype=torch.float32,
        device=device,
    )

    epochs = 1 if args.smoke else args.epochs
    limit = 2 if args.smoke else None

    initial_weight = (
        model.encoder[0].weight.detach().cpu().clone()
        if args.smoke
        else None
    )

    best_loss = float("inf")
    best_weights = None
    best_epoch = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_n = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            positive_weight,
            limit,
        )

        val_loss, metrics = validate(
            model,
            val_loader,
            device,
            positive_weight,
            limit,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": metrics,
        })

        print(
            f"EPOCH {epoch:03d}:"
            f" train_loss={train_loss:.6f}"
            f" val_loss={val_loss:.6f}"
            f" train_n={train_n}"
            f" val_n={metrics['overall']['n']}"
        )

        best_loss, bad_epochs, improved, should_stop = (
            early_stopping_update(
                val_loss, best_loss, bad_epochs, args.patience
            )
        )

        if improved:
            best_epoch = epoch

            if not args.smoke:
                best_weights = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor
                    in model.state_dict().items()
                }

        if not args.smoke and should_stop:
            print(
                f"EARLY STOPPING: epoch={epoch}"
                f" best_epoch={best_epoch}"
                f" best_val_loss={best_loss:.6f}"
                f" patience={args.patience}",
                flush=True,
            )
            break

    if args.smoke:
        assert not torch.equal(
            initial_weight,
            model.encoder[0].weight.detach().cpu(),
        ), "Optimizer did not update parameters"

        assert metrics["overall"]["n"] == min(
            2 * args.batch_size,
            len(val_data),
        )

        print(
            "PASS: optimizer updated actual"
            " visual encoder parameters"
        )
        print(
            "PASS: train/val loss and finite"
            " physics/contact metrics"
        )
        print(
            "PASS: no filesystem outputs during smoke"
        )
        print(
            "M4 TRAINING PIPELINE CPU SMOKE: PASSED"
        )
        return

    assert best_weights is not None
    assert best_epoch is not None
    assert 1 <= len(history) <= args.epochs

    OUT.mkdir(parents=True, exist_ok=True)

    checkpoint = OUT / "best_model.pt"
    report_path = OUT / "training_history.json"

    config = {
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "train_contact_weight": balance,
        "train_contact_sampling_weight": 3.0,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    payload = {
        "state_dict": best_weights,
        "model": "VisualPhysicalWorldModel",
        "parameter_count": model.parameter_count(),
        "best_val_loss": best_loss,
        "config": config,
        "source_sha256": SOURCE_HASHES,
    }

    save_new_atomic(
        checkpoint,
        lambda path: torch.save(payload, path),
    )

    report = {
        "schema_version": "m4-visual-training-v0.1",
        "config": config,
        "best_val_loss": best_loss,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": SOURCE_HASHES,
        "evaluation_policy": (
            "Train for optimization; validation for checkpoint;"
            " held-out test untouched"
        ),
    }

    save_new_atomic(
        report_path,
        lambda path: path.write_text(
            json.dumps(
                report,
                indent=2,
                allow_nan=False,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        ),
    )

    print("BEST CHECKPOINT:", checkpoint)
    print("CHECKPOINT SHA256:", sha256(checkpoint))
    print("TRAINING REPORT:", report_path)
    print("M4 FULL TRAINING: PASSED")


if __name__ == "__main__":
    main()
