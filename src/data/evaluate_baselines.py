"""Evaluate deterministic one-step PushT state prediction baselines."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/m3_pilot_v01"
PROCESSED = ROOT / "data/processed/m3_pilot_v01"

SPLIT_FILE = PROCESSED / "splits.json"
MANIFEST_FILE = RAW / "manifest.json"
OUTPUT_FILE = PROCESSED / "baseline_metrics.json"

MODES = ("center_push", "offset_push", "free_motion")
SPLITS = ("train", "val", "test")
BASELINES = ("persistence", "constant_velocity")

EXPECTED_FILES = {
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
    "src/data/generate_pilot.py":
        "d2466de34c5303ca86fed608d389d9dc272ada6355f767514e9aaf3321e22eba",
    "src/data/split_pilot.py":
        "f27c572111fd1fe3d78e070bd1c93f3d001616bc348a8f533baedf9a9995ab84",
}

EXPECTED_SPLIT_SHA256 = (
    "16c63b57d09c03b7b5d52f7d7018938fa5024b88da4136aef34c16e36f9e0cc6"
)

DT = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angular_difference(a: float, b: float) -> float:
    """Signed shortest angular difference in radians."""
    delta = a - b
    return math.atan2(math.sin(delta), math.cos(delta))


def predict(state: np.ndarray, baseline: str) -> tuple[np.ndarray, float]:
    """Use only the current state to predict the next block pose."""
    xy = state[3:5].copy()
    yaw = float(state[6])

    if baseline == "constant_velocity":
        # For the free joint, qvel[0:2] is translational velocity.
        xy += DT * state[7:9]

        # qvel[5] is the body-frame z angular velocity.
        # This is a planar-yaw approximation.
        yaw += DT * float(state[12])

    elif baseline != "persistence":
        raise ValueError(f"Unknown baseline: {baseline}")

    return xy, yaw


def summarize(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    pos = np.array(
        [r["position_error_m"] for r in rows],
        dtype=np.float64,
    )
    yaw = np.array(
        [r["yaw_error_rad"] for r in rows],
        dtype=np.float64,
    )

    assert np.isfinite(pos).all()
    assert np.isfinite(yaw).all()

    return {
        "transitions": len(rows),
        "position_mean_l2_cm": float(
            np.mean(pos) * 100.0
        ),
        "position_rmse_l2_cm": float(
            np.sqrt(np.mean(pos ** 2)) * 100.0
        ),
        "yaw_mae_deg": float(
            np.degrees(np.mean(yaw))
        ),
        "yaw_rmse_deg": float(
            np.degrees(np.sqrt(np.mean(yaw ** 2)))
        ),
    }


def validate_inputs() -> tuple[dict, dict]:
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT_FILE}"
        )

    for filename, expected_hash in EXPECTED_FILES.items():
        assert sha256(ROOT / filename) == expected_hash, (
            f"Source provenance mismatch: {filename}"
        )

    assert sha256(SPLIT_FILE) == EXPECTED_SPLIT_SHA256

    manifest = json.loads(
        MANIFEST_FILE.read_text(encoding="utf-8")
    )
    splits = json.loads(
        SPLIT_FILE.read_text(encoding="utf-8")
    )

    assert splits["source_manifest_sha256"] == (
        sha256(MANIFEST_FILE)
    )
    assert splits["unit_of_split"] == "complete_episode"
    assert splits["steps_per_episode"] == 64

    assert manifest["episodes"] == 12
    assert manifest["steps_per_episode"] == 64
    assert math.isclose(manifest["control_dt"], DT)

    assert set(splits["splits"]) == set(SPLITS)

    manifest_items = {
        item["file"]: item
        for item in manifest["items"]
    }
    assert len(manifest_items) == 12

    expected_counts = {
        "train": 6,
        "val": 3,
        "test": 3,
    }
    expected_by_mode = {
        "train": 2,
        "val": 1,
        "test": 1,
    }

    seen = set()

    for split in SPLITS:
        records = splits["splits"][split]
        assert len(records) == expected_counts[split]

        for mode in MODES:
            assert sum(
                item["mode"] == mode
                for item in records
            ) == expected_by_mode[split]

        for item in records:
            filename = item["file"]
            assert filename not in seen
            seen.add(filename)

            assert filename in manifest_items
            assert item["mode"] == manifest_items[filename]["mode"]
            assert item["sha256"] == manifest_items[filename]["sha256"]
            assert sha256(RAW / filename) == item["sha256"]

    assert seen == set(manifest_items)

    print("PASS: source and dataset provenance")
    print("PASS: disjoint stratified episode splits")

    return manifest, splits


def evaluate(splits: dict) -> dict:
    metrics = {}

    for split in SPLITS:
        all_rows = {
            baseline: []
            for baseline in BASELINES
        }

        for item in splits["splits"][split]:
            mode = item["mode"]

            with np.load(
                RAW / item["file"],
                allow_pickle=False,
            ) as data:
                states = data["states"]
                actions = data["actions"]
                rgb = data["rgb"]

                assert states.shape == (65, 14)
                assert actions.shape == (64, 2)
                assert rgb.shape == (65, 64, 64, 3)
                assert np.isfinite(states).all()
                assert np.isfinite(actions).all()

                assert np.allclose(
                    states[:, 0],
                    np.arange(65) * DT,
                    atol=1e-10,
                    rtol=0,
                )

                for t in range(64):
                    current = states[t]
                    target = states[t + 1]

                    target_xy = target[3:5]
                    target_yaw = float(target[6])

                    next_contact = float(target[13])
                    assert next_contact in (0.0, 1.0)

                    for baseline in BASELINES:
                        prediction_xy, prediction_yaw = predict(
                            current, baseline
                        )

                        position_error = float(
                            np.linalg.norm(
                                prediction_xy - target_xy
                            )
                        )

                        yaw_error = abs(
                            angular_difference(
                                prediction_yaw,
                                target_yaw,
                            )
                        )

                        all_rows[baseline].append({
                            "mode": mode,
                            "next_state_contact": bool(next_contact),
                            "position_error_m": position_error,
                            "yaw_error_rad": yaw_error,
                        })

        metrics[split] = {}

        for baseline in BASELINES:
            rows = all_rows[baseline]
            assert len(rows) == (
                len(splits["splits"][split]) * 64
            )

            metrics[split][baseline] = {
                "overall": summarize(rows),
                "by_mode": {
                    mode: summarize([
                        row for row in rows
                        if row["mode"] == mode
                    ])
                    for mode in MODES
                },
                "by_next_state_contact": {
                    "contact": summarize([
                        row for row in rows
                        if row["next_state_contact"]
                    ]),
                    "no_contact": summarize([
                        row for row in rows
                        if not row["next_state_contact"]
                    ]),
                },
            }

    return metrics


def main() -> None:
    manifest, splits = validate_inputs()

    result_a = evaluate(splits)
    result_b = evaluate(splits)

    assert result_a == result_b, (
        "Baseline evaluation is not repeatable"
    )
    print("PASS: deterministic baseline evaluation")

    report = {
        "schema_version": "m3-baselines-v0.1",
        "dataset": "m3_pilot_v01",
        "source_manifest_sha256": sha256(MANIFEST_FILE),
        "source_splits_sha256": sha256(SPLIT_FILE),
        "seed": manifest["seed"],
        "prediction_horizon_steps": 1,
        "control_dt_seconds": DT,
        "target": "next T-block XY position and planar yaw",
        "input_policy": "current state only; no future actions or states",
        "baselines": {
            "persistence":
                "Next block XY and yaw equal current block XY and yaw.",
            "constant_velocity":
                "Extrapolate current XY velocity and body z angular "
                "velocity for 0.05 seconds; planar approximation.",
        },
        "contact_grouping":
            "Post-hoc evaluation only; uses NEXT state contact "
            "labels, never baseline prediction inputs.",
        "metrics": result_a,
    }

    for split in SPLITS:
        print(f"SPLIT {split.upper()}")

        for baseline in BASELINES:
            entry = result_a[split][baseline]
            overall = entry["overall"]

            print(
                f"  BASELINE {baseline}"
                f" transitions={overall['transitions']}"
                f" position_mean_l2_cm="
                f"{overall['position_mean_l2_cm']:.6f}"
                f" yaw_mae_deg={overall['yaw_mae_deg']:.6f}"
            )

            for mode in MODES:
                metric = entry["by_mode"][mode]

                print(
                    f"    MODE {mode}"
                    f" n={metric['transitions']}"
                    f" pos_cm={metric['position_mean_l2_cm']:.6f}"
                    f" yaw_deg={metric['yaw_mae_deg']:.6f}"
                )

            for group in ("contact", "no_contact"):
                metric = entry["by_next_state_contact"][group]

                if metric is None:
                    print(f"    GROUP {group} n=0")
                else:
                    print(
                        f"    GROUP {group}"
                        f" n={metric['transitions']}"
                        f" pos_cm={metric['position_mean_l2_cm']:.6f}"
                        f" yaw_deg={metric['yaw_mae_deg']:.6f}"
                    )

    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    OUTPUT_FILE.write_text(
        serialized,
        encoding="utf-8",
    )

    print("BASELINE FILE:", OUTPUT_FILE.resolve())
    print("BASELINE FILE SHA256:", sha256(OUTPUT_FILE))
    print("M3 PILOT BASELINE EVALUATION: PASSED")


if __name__ == "__main__":
    main()
