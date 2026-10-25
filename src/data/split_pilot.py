"""Deterministic stratified episode-level split for the M3 pilot."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/m3_pilot_v01"
OUTPUT = ROOT / "data/processed/m3_pilot_v01"
SEED = 20261025

MODES = ("center_push", "offset_push", "free_motion")

COUNTS_PER_MODE = {
    "train": 2,
    "val": 1,
    "test": 1,
}

EXPECTED_SOURCE_HASHES = {
    "src/data/generate_pilot.py":
        "d2466de34c5303ca86fed608d389d9dc272ada6355f767514e9aaf3321e22eba",
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_key(mode: str, filename: str) -> str:
    """Stable across Python processes and NumPy versions."""
    payload = f"{SEED}|{mode}|{filename}"
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    for name, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256(ROOT / name)
        assert actual == expected, (
            f"Unexpected source version: {name}"
        )

    print("PASS: source provenance")

    manifest_path = RAW / "manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "m3-pilot-v0.1"
    assert manifest["seed"] == SEED
    assert manifest["episodes"] == 12
    assert manifest["steps_per_episode"] == 64
    assert manifest["control_dt"] == 0.05

    assert manifest["generator_sha256"] == (
        EXPECTED_SOURCE_HASHES["src/data/generate_pilot.py"]
    )

    assert manifest["environment_sha256"] == (
        EXPECTED_SOURCE_HASHES["src/simulation/pusht_env.py"]
    )

    assert manifest["scene_sha256"] == (
        EXPECTED_SOURCE_HASHES["src/simulation/pusht_scene.xml"]
    )

    items = manifest["items"]
    assert len(items) == 12

    expected_files = {
        f"episode_{i:03d}.npz" for i in range(12)
    }

    actual_files = {
        path.name for path in RAW.glob("*.npz")
    }

    assert actual_files == expected_files
    assert {item["file"] for item in items} == expected_files
    assert len({item["file"] for item in items}) == 12

    grouped = {mode: [] for mode in MODES}

    for item in items:
        filename = item["file"]
        mode = item["mode"]

        assert mode in MODES

        path = RAW / filename
        assert sha256(path) == item["sha256"]

        with np.load(path, allow_pickle=False) as data:
            states = data["states"]
            actions = data["actions"]
            rgb = data["rgb"]

            assert states.shape == (65, 14)
            assert actions.shape == (64, 2)
            assert rgb.shape == (65, 64, 64, 3)
            assert rgb.dtype == np.uint8
            assert np.isfinite(states).all()
            assert np.isfinite(actions).all()

            assert np.allclose(
                states[:, 0],
                np.arange(65) * 0.05,
                atol=1e-10,
                rtol=0,
            )

            assert np.all(
                np.linalg.norm(actions, axis=1)
                <= 0.020000000001
            )

            assert int(states[1:, 13].sum()) == (
                item["contact_observations"]
            )

        grouped[mode].append({
            "file": filename,
            "mode": mode,
            "sha256": item["sha256"],
        })

    assert all(
        len(grouped[mode]) == 4 for mode in MODES
    )

    print("PASS: all 12 episode files verified")

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for mode in MODES:
        ranked = sorted(
            grouped[mode],
            key=lambda item: (
                ranking_key(mode, item["file"]),
                item["file"],
            ),
        )

        splits["train"].extend(ranked[:2])
        splits["val"].append(ranked[2])
        splits["test"].append(ranked[3])

    expected_counts = {
        "train": 6,
        "val": 3,
        "test": 3,
    }

    all_assigned = []

    for split_name, records in splits.items():
        assert len(records) == expected_counts[split_name]

        counts = Counter(
            item["mode"] for item in records
        )

        assert all(
            counts[mode] == COUNTS_PER_MODE[split_name]
            for mode in MODES
        )

        all_assigned.extend(
            item["file"] for item in records
        )

    assert len(all_assigned) == 12
    assert len(set(all_assigned)) == 12
    assert set(all_assigned) == expected_files

    print("PASS: disjoint episode-level splits")
    print("PASS: balanced mode coverage")

    result = {
        "schema_version": "m3-pilot-split-v0.1",
        "dataset": "m3_pilot_v01",
        "source_manifest": "data/raw/m3_pilot_v01/manifest.json",
        "source_manifest_sha256": sha256(manifest_path),
        "seed": SEED,
        "method": "per-mode deterministic SHA256 ranking",
        "unit_of_split": "complete_episode",
        "steps_per_episode": 64,
        "counts_per_mode": COUNTS_PER_MODE,
        "splits": splits,
    }

    # Only create output after all validation succeeds.
    OUTPUT.mkdir(parents=True, exist_ok=False)

    output_file = OUTPUT / "splits.json"

    output_file.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("SPLIT ASSIGNMENTS")

    for split_name, records in splits.items():
        print(
            f"{split_name.upper()}:"
            f" episodes={len(records)}"
            f" transitions={len(records) * 64}"
        )

        for item in records:
            print(
                f"  {item['file']} mode={item['mode']}"
            )

    print("SPLIT FILE:", output_file.resolve())
    print("SPLIT FILE SHA256:", sha256(output_file))
    print("M3 PILOT EPISODE SPLIT: PASSED")


if __name__ == "__main__":
    main()
