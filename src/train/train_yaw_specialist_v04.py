"""M4 V0.4: reproducible silhouette-only yaw specialist experiment.

Current RGB(t) is the only inference input. Training labels are used solely
by the loss and the evaluator. Test data are never opened in this module.
"""

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
from torch.utils.data import DataLoader

from src.data.world_model_dataset import ActionWorldWindowDataset
from src.models.yaw_specialist_v04 import SilhouetteYawSpecialist

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/processed/m4_yaw_v04"
MODES = ("center_push", "offset_push", "free_motion")
EXPECTED_HASHES = {
    "src/data/world_model_dataset.py": "0c30473110082e2f0bc851b8f03f9a4f7c9f13dec917cfb3606022aebce96ab3",
    "src/models/visual_dynamics.py": "665487d0b6e2e979727601e0d25ecf988345a03059abe3865e06a3fa76e7e57a",
    "src/models/visual_dynamics_v03.py": "d50c8debb7b04477861d248e4c6e5ec87ca63b247f48231bb4a9aa198619a659",
    "src/models/yaw_specialist_v04.py": "972b6067932a1b3de3f99f6dbb4d63129aae8c14daf24d4eacc9a6c6db35509c",
    "src/train/train_visual_dynamics.py": "b58eebfa61764dd98b6d6a743beadd6a2ae9ea6244e6dedfcc16620213d38edc",
    "src/train/train_visual_dynamics_v02.py": "e2c0f003ad956c39ba198f11b6fc5f5e06cf038e9e5a662d15a4eacd9e872963",
    "src/train/train_visual_dynamics_v03.py": "11e1c366d5b23df7abf6feda4f48bc0c2961c2c7221fcb49812ab5b8acdcf570",
    "data/raw/m3_main_v01/manifest.json": "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50",
    "data/processed/m3_main_v01/splits.json": "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa",
    "data/processed/m4_visual_v01/best_model.pt": "3af77d096be257f3b0f806474a3995ead252f4bc0fb25e4543f248dfd59926d5",
    "data/processed/m4_visual_v01/training_history.json": "234bf0a9b24acb15e44debb63a8a39c6abd1e7b3cf93bc1c90681c79256bdaf4",
    "data/processed/m4_visual_v02/best_model.pt": "2b0417fdefcc188b4852bbc7c472c3cf3136ddde00b70e621c3c074db2d65463",
    "data/processed/m4_visual_v02/training_history.json": "546184c1d0824075a7123c5f6b7993594afc90e7afc5c507e57f0932a1890ba2",
    "data/processed/m4_visual_v03/best_model.pt": "d0e24224680e557b2619595a3adcfc09183d38bc43a22cb699a074df5f8a95bd",
    "data/processed/m4_visual_v03/training_history.json": "6a6c0c80869c4b0166bff0717658e94a39629eb8f4fa71a90578920465886144",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative
    print("PASS: 15 frozen sources/dataset/artifacts")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_loss(predicted, target):
    assert predicted.shape == target.shape
    assert predicted.ndim == 2 and predicted.shape[1] == 2
    return (1.0 - (predicted * target).sum(dim=1)).mean()


def move_batch(batch, device):
    model_input = batch["model_input"]
    assert set(model_input) == {"frames", "action"}
    return (
        model_input["frames"].to(device=device),
        batch["labels"]["current_yaw_cs"].to(device=device),
    )


def train_epoch(model, loader, optimizer, device, *, limit_batches=None):
    model.train()
    total = 0.0
    count = 0
    batches = 0
    for batch in loader:
        frames, target = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(frames)["current_yaw_cs"]
        loss = cosine_loss(prediction, target)
        assert torch.isfinite(loss).item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        n = int(frames.shape[0])
        total += float(loss.detach().cpu()) * n
        count += n
        batches += 1
        if limit_batches is not None and batches >= limit_batches:
            break
    assert count > 0
    return {"loss": total / count, "n": count, "batches": batches}


def describe(errors):
    assert errors
    array = np.asarray(errors, dtype=np.float64)
    assert np.isfinite(array).all()
    return {
        "n": int(array.size),
        "mae_deg": float(array.mean()),
        "median_deg": float(np.median(array)),
        "p90_deg": float(np.percentile(array, 90)),
        "ge90_fraction": float(np.mean(array >= 90.0)),
    }


def evaluate(model, loader, device, *, limit_batches=None, full=False):
    model.eval()
    groups = {"overall": []}
    groups.update({mode: [] for mode in MODES})
    total_loss = 0.0
    count = 0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            frames, target = move_batch(batch, device)
            prediction = model(frames)["current_yaw_cs"]
            loss = cosine_loss(prediction, target)
            assert torch.isfinite(loss).item()
            n = int(frames.shape[0])
            total_loss += float(loss.detach().cpu()) * n
            count += n
            cross = prediction[:, 0] * target[:, 1] - prediction[:, 1] * target[:, 0]
            dot = (prediction * target).sum(dim=1)
            errors = (torch.atan2(cross.abs(), dot) * (180.0 / math.pi)).detach().cpu().tolist()
            modes = batch["metadata"]["mode"]
            assert len(modes) == n
            for mode, error in zip(modes, errors):
                assert mode in MODES
                assert math.isfinite(error)
                groups["overall"].append(float(error))
                groups[mode].append(float(error))
            batches += 1
            if limit_batches is not None and batches >= limit_batches:
                break
    assert count == len(groups["overall"]) and count > 0
    result = {
        "val_loss": total_loss / count,
        "n": count,
        "groups": {name: describe(values) for name, values in groups.items() if values},
    }
    if full:
        assert count == 1701
        assert all(len(groups[mode]) == 567 for mode in MODES)
        assert set(result["groups"]) == {"overall", *MODES}
    return result


def atomic_write(path, kind, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix="tmp_m4_yaw_", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        if kind == "torch":
            torch.save(content, temporary)
        elif kind == "json":
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
        else:
            raise ValueError(kind)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    assert args.epochs >= 1 and args.patience >= 1
    assert args.batch_size >= 2 and 0 < args.lr <= 0.01
    if args.smoke:
        assert args.device == "cpu", "smoke test must use CPU"
    else:
        assert not (OUTPUT / "best_model.pt").exists()
        assert not (OUTPUT / "training_history.json").exists()
    verify_frozen()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS unavailable; do not fallback"
    train_dataset = ActionWorldWindowDataset("train")
    val_dataset = ActionWorldWindowDataset("val")
    assert len(train_dataset) == 7938 and len(val_dataset) == 1701
    print("TRAIN=7938 VAL=1701 TEST=UNOPENED")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    model = SilhouetteYawSpecialist().to(device)
    assert sum(p.numel() for p in model.parameters()) == 231362
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = None
    best_epoch = 0
    best_score = float("inf")
    missed = 0
    epoch_limit = 1 if args.smoke else args.epochs
    for epoch in range(1, epoch_limit + 1):
        training = train_epoch(model, train_loader, optimizer, device,
                               limit_batches=2 if args.smoke else None)
        validation = evaluate(model, val_loader, device,
                              limit_batches=2 if args.smoke else None,
                              full=not args.smoke)
        score = float(validation["groups"]["overall"]["mae_deg"])
        row = {"epoch": epoch, "train": training, "val": validation,
               "selection_score_yaw_mae_deg": score}
        history.append(row)
        print(f"EPOCH {epoch:03}: train_loss={training['loss']:.6f} "
              f"val_loss={validation['val_loss']:.6f} "
              f"yaw_mae={score:.6f} val_n={validation['n']}", flush=True)
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            missed = 0
        else:
            missed += 1
        if missed >= args.patience:
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}")
            break
    assert best is not None and best_epoch >= 1
    if args.smoke:
        assert history[0]["train"]["n"] <= args.batch_size * 2
        assert history[0]["val"]["n"] <= args.batch_size * 2
        assert not (OUTPUT / "best_model.pt").exists()
        assert not (OUTPUT / "training_history.json").exists()
        print("M4 V04 YAW TRAINER CPU SMOKE: PASSED")
        return
    assert history[best_epoch - 1]["val"]["n"] == 1701
    assert all(row["val"]["n"] == 1701 and row["train"]["n"] == 7938 for row in history)
    config = {"device": str(device), "epochs": args.epochs, "patience": args.patience,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    checkpoint = {
        "schema": "m4-yaw-specialist-v04",
        "model_name": "SilhouetteYawSpecialist",
        "parameter_count": 231362,
        "state_dict": best,
        "best_epoch": best_epoch,
        "best_val_yaw_mae_deg": best_score,
        "config": config,
        "source_sha256": sha256(Path(__file__)),
        "dataset_sha256": {name: sha256(ROOT / name) for name in
                           ("data/raw/m3_main_v01/manifest.json", "data/processed/m3_main_v01/splits.json")},
    }
    checkpoint_path = OUTPUT / "best_model.pt"
    history_path = OUTPUT / "training_history.json"
    atomic_write(checkpoint_path, "torch", checkpoint)
    digest = sha256(checkpoint_path)
    report = {
        "schema": "m4-yaw-specialist-training-v04",
        "config": config,
        "source_sha256": checkpoint["source_sha256"],
        "checkpoint_sha256": digest,
        "best_epoch": best_epoch,
        "best_val_yaw_mae_deg": best_score,
        "best_val_metrics": history[best_epoch - 1]["val"],
        "history": history,
        "train_windows": 7938,
        "val_windows": 1701,
        "test": "unopened",
    }
    atomic_write(history_path, "json", report)
    print("CHECKPOINT:", checkpoint_path)
    print("CHECKPOINT SHA256:", digest)
    print("HISTORY:", history_path)
    print("M4 V04 YAW FULL TRAINING: PASSED")


if __name__ == "__main__":
    main()
