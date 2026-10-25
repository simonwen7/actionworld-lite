"""Evaluate one-step privileged-state baselines on the M3 main dataset."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data import evaluate_baselines as reference


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/m3_main_v01"
PROCESSED = ROOT / "data/processed/m3_main_v01"
MANIFEST_FILE = RAW / "manifest.json"
SPLIT_FILE = PROCESSED / "splits.json"
OUTPUT_FILE = PROCESSED / "baseline_metrics.json"

MODES = ("center_push", "offset_push", "free_motion")
SPLITS = ("train", "val", "test")
BASELINES = ("persistence", "constant_velocity")
DT = 0.05

EXPECTED_HASHES = {
    "src/data/generate_main.py":
        "4588785fd5deec05d877d4154352b6c910ecd91cb11860ccdc5d67e3305de8d8",
    "src/data/split_main.py":
        "91b0188b7ab7a9eb2a9842b5d212967f654316eb1528776ecf17ad373332d1fc",
    "src/data/evaluate_baselines.py":
        "35e9cfd03d25bb6f3f130a6196c7030baf29297a74fa8c371fad2140254206c4",
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
}

EXPECTED_MANIFEST_SHA256 = (
    "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50"
)
EXPECTED_SPLIT_SHA256 = (
    "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs() -> dict:
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_FILE}")

    for filename, expected in EXPECTED_HASHES.items():
        if sha256(ROOT / filename) != expected:
            raise RuntimeError(f"Source hash mismatch: {filename}")

    assert sha256(MANIFEST_FILE) == EXPECTED_MANIFEST_SHA256
    assert sha256(SPLIT_FILE) == EXPECTED_SPLIT_SHA256
    assert reference.DT == DT
    assert reference.BASELINES == BASELINES

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    split_info = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "m3-main-v0.1"
    assert manifest["episodes"] == 180
    assert manifest["steps_per_episode"] == 64
    assert math.isclose(manifest["control_dt"], DT)
    assert manifest["image_shape"] == [64, 64, 3]
    assert split_info["schema_version"] == "m3-main-split-v0.1"
    assert split_info["source_manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert split_info["unit_of_split"] == "complete_episode"
    assert split_info["steps_per_episode"] == 64
    assert math.isclose(split_info["control_dt"], DT)
    assert set(split_info["splits"]) == set(SPLITS)

    source = {item["file"]: item for item in manifest["items"]}
    assert len(source) == 180
    assert len(manifest["items"]) == 180

    expected_counts = {"train": 126, "val": 27, "test": 27}
    expected_per_mode = {"train": 42, "val": 9, "test": 9}
    seen = set()

    for split in SPLITS:
        entries = split_info["splits"][split]
        assert len(entries) == expected_counts[split]
        counts = Counter(item["mode"] for item in entries)
        assert all(counts[mode] == expected_per_mode[split] for mode in MODES)

        for entry in entries:
            filename = entry["file"]
            assert filename not in seen
            assert filename in source
            seen.add(filename)
            assert entry["mode"] == source[filename]["mode"]
            assert entry["episode_seed"] == source[filename]["episode_seed"]
            assert entry["sha256"] == source[filename]["sha256"]
            assert sha256(RAW / filename) == entry["sha256"]

    assert seen == set(source)
    print("PASS: source, dataset, and split provenance")
    print("PASS: 180 disjoint mode-balanced episodes")
    return split_info


def evaluate(split_info: dict) -> dict:
    result = {}

    for split in SPLITS:
        rows = {baseline: [] for baseline in BASELINES}

        for entry in split_info["splits"][split]:
            with np.load(RAW / entry["file"], allow_pickle=False) as archive:
                states = archive["states"]
                actions = archive["actions"]
                rgb = archive["rgb"]

                assert states.shape == (65, 14)
                assert actions.shape == (64, 2)
                assert rgb.shape == (65, 64, 64, 3)
                assert rgb.dtype == np.uint8
                assert np.isfinite(states).all()
                assert np.isfinite(actions).all()
                assert np.allclose(
                    states[:, 0], np.arange(65) * DT,
                    atol=1e-10, rtol=0,
                )
                assert np.all(
                    np.linalg.norm(actions, axis=1) <= 0.020000000001
                )

                for t in range(64):
                    current = states[t]
                    target = states[t + 1]
                    next_contact = float(target[13])
                    assert next_contact in (0.0, 1.0)

                    for baseline in BASELINES:
                        predicted_xy, predicted_yaw = reference.predict(
                            current, baseline
                        )
                        pos_error = float(np.linalg.norm(
                            predicted_xy - target[3:5]
                        ))
                        yaw_error = abs(reference.angular_difference(
                            predicted_yaw, float(target[6])
                        ))
                        rows[baseline].append({
                            "mode": entry["mode"],
                            "next_state_contact": bool(next_contact),
                            "position_error_m": pos_error,
                            "yaw_error_rad": yaw_error,
                        })

        result[split] = {}
        for baseline in BASELINES:
            samples = rows[baseline]
            assert len(samples) == len(split_info["splits"][split]) * 64
            result[split][baseline] = {
                "overall": reference.summarize(samples),
                "by_mode": {
                    mode: reference.summarize([
                        row for row in samples if row["mode"] == mode
                    ]) for mode in MODES
                },
                "by_next_state_contact": {
                    "contact": reference.summarize([
                        row for row in samples if row["next_state_contact"]
                    ]),
                    "no_contact": reference.summarize([
                        row for row in samples if not row["next_state_contact"]
                    ]),
                },
            }

    return result


def main() -> None:
    split_info = validate_inputs()
    metrics = evaluate(split_info)
    assert metrics == evaluate(split_info)
    print("PASS: deterministic baseline evaluation")

    report = {
        "schema_version": "m3-main-baselines-v0.1",
        "dataset": "m3_main_v01",
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_splits_sha256": EXPECTED_SPLIT_SHA256,
        "prediction_horizon_steps": 1,
        "control_dt_seconds": DT,
        "target": "next T-block XY position and planar yaw",
        "input_policy": "current simulator state only; no future states/actions",
        "privileged_state_baseline": True,
        "visual_only_model_comparison_is_input_mismatched": True,
        "contact_grouping": "next-state contact, used for scoring only",
        "metrics": metrics,
    }

    for split in SPLITS:
        print(f"SPLIT {split.upper()}")
        for baseline in BASELINES:
            summary = metrics[split][baseline]
            overall = summary["overall"]
            print(
                f"  {baseline} n={overall['transitions']}"
                f" pos_cm={overall['position_mean_l2_cm']:.6f}"
                f" yaw_deg={overall['yaw_mae_deg']:.6f}"
            )
            for mode in MODES:
                item = summary["by_mode"][mode]
                print(
                    f"    mode={mode} n={item['transitions']}"
                    f" pos_cm={item['position_mean_l2_cm']:.6f}"
                    f" yaw_deg={item['yaw_mae_deg']:.6f}"
                )
            for label in ("contact", "no_contact"):
                item = summary["by_next_state_contact"][label]
                if item is None:
                    print(f"    {label}: n=0")
                else:
                    print(
                        f"    {label} n={item['transitions']}"
                        f" pos_cm={item['position_mean_l2_cm']:.6f}"
                        f" yaw_deg={item['yaw_mae_deg']:.6f}"
                    )

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("BASELINE OUTPUT:", OUTPUT_FILE.resolve())
    print("BASELINE SHA256:", sha256(OUTPUT_FILE))
    print("M3 MAIN BASELINES: PASSED")


if __name__ == "__main__":
    main()
