"""Reproducible stratified episode split for ActionWorld Lite M3."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/m3_main_v01"
OUTPUT = ROOT / "data/processed/m3_main_v01"
SPLIT_FILE = OUTPUT / "splits.json"

SEED = 20261026
EPISODES = 180
STEPS = 64
DT = 0.05

MODES = ("center_push", "offset_push", "free_motion")

COUNTS_PER_MODE = {
    "train": 42,
    "val": 9,
    "test": 9,
}

EXPECTED_MANIFEST_SHA256 = (
    "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50"
)

EXPECTED_SOURCE_HASHES = {
    "src/data/generate_main.py":
        "4588785fd5deec05d877d4154352b6c910ecd91cb11860ccdc5d67e3305de8d8",
    "src/data/generate_pilot.py":
        "d2466de34c5303ca86fed608d389d9dc272ada6355f767514e9aaf3321e22eba",
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_key(mode: str, filename: str) -> tuple[str, str]:
    """Stable ranking independent of process-specific hash seeds."""
    payload = f"{SEED}|m3-main-v0.1|{mode}|{filename}"
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    return digest, filename


def verify_sources() -> None:
    for filename, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256(ROOT / filename)
        if actual != expected:
            raise RuntimeError(
                f"Source integrity mismatch: {filename}"
            )

    manifest_hash = sha256(RAW / "manifest.json")

    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Main Dataset manifest SHA256 mismatch"
        )

    print("PASS: source and manifest provenance")


def load_and_verify_episodes() -> tuple[dict, dict]:
    manifest = json.loads(
        (RAW / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "m3-main-v0.1"
    assert manifest["seed"] == SEED
    assert manifest["episodes"] == EPISODES
    assert manifest["steps_per_episode"] == STEPS
    assert math.isclose(manifest["control_dt"], DT)
    assert manifest["image_shape"] == [64, 64, 3]
    assert len(manifest["state_fields"]) == 14

    assert manifest["generator_sha256"] == (
        EXPECTED_SOURCE_HASHES["src/data/generate_main.py"]
    )

    for filename, expected in manifest["source_sha256"].items():
        assert filename in EXPECTED_SOURCE_HASHES
        assert expected == EXPECTED_SOURCE_HASHES[filename]

    assert len(manifest["source_sha256"]) == 3

    items = manifest["items"]
    assert len(items) == EPISODES

    expected_files = {
        f"episode_{i:03d}.npz"
        for i in range(EPISODES)
    }

    existing_files = {
        p.name for p in RAW.glob("*.npz")
    }

    assert existing_files == expected_files

    grouped = {
        mode: []
        for mode in MODES
    }

    for i, item in enumerate(items):
        filename = f"episode_{i:03d}.npz"
        mode = MODES[i % len(MODES)]
        episode_seed = SEED + i * 1009

        assert item["file"] == filename
        assert item["mode"] == mode
        assert item["episode_seed"] == episode_seed

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
                np.arange(65) * DT,
                atol=1e-10,
                rtol=0,
            )

            assert np.all(
                np.linalg.norm(actions, axis=1)
                <= 0.020000000001
            )

            contacts = states[1:, 13]

            assert np.all(
                (contacts == 0) | (contacts == 1)
            )

            assert int(contacts.sum()) == (
                item["contact_observations"]
            )

            assert np.allclose(
                states[0, 3:5],
                item["initial"]["block_xy"],
                atol=1e-9,
                rtol=0,
            )

            assert np.allclose(
                states[0, 1:3],
                item["initial"]["pusher_xy"],
                atol=1e-9,
                rtol=0,
            )

        grouped[mode].append({
            "file": filename,
            "mode": mode,
            "episode_seed": episode_seed,
            "sha256": item["sha256"],
        })

    assert all(
        len(grouped[mode]) == 60
        for mode in MODES
    )

    print("PASS: 180 verified episode archives")
    print("PASS: state, action, and RGB shapes")
    print("PASS: episode metadata and temporal alignment")

    return manifest, grouped


def build_splits(grouped: dict) -> dict:
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for mode in MODES:
        ranked = sorted(
            grouped[mode],
            key=lambda item: rank_key(
                mode, item["file"]
            ),
        )

        splits["train"].extend(ranked[:42])
        splits["val"].extend(ranked[42:51])
        splits["test"].extend(ranked[51:60])

    expected_totals = {
        "train": 126,
        "val": 27,
        "test": 27,
    }

    seen = set()

    for split_name, records in splits.items():
        assert len(records) == expected_totals[split_name]

        counts = Counter(
            item["mode"] for item in records
        )

        for mode in MODES:
            assert counts[mode] == (
                COUNTS_PER_MODE[split_name]
            )

        for item in records:
            assert item["file"] not in seen
            seen.add(item["file"])

    assert len(seen) == EPISODES

    print("PASS: disjoint complete-episode assignments")
    print("PASS: 70/15/15 stratified mode coverage")

    return splits


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {OUTPUT}"
        )

    verify_sources()

    manifest, grouped = load_and_verify_episodes()
    splits = build_splits(grouped)

    report = {
        "schema_version": "m3-main-split-v0.1",
        "dataset": "m3_main_v01",
        "source_manifest":
            "data/raw/m3_main_v01/manifest.json",
        "source_manifest_sha256":
            EXPECTED_MANIFEST_SHA256,
        "source_generator_sha256":
            EXPECTED_SOURCE_HASHES[
                "src/data/generate_main.py"
            ],
        "seed": SEED,
        "method":
            "per-mode deterministic SHA256 ranking",
        "ranking_format":
            "{seed}|m3-main-v0.1|{mode}|{filename}",
        "unit_of_split": "complete_episode",
        "steps_per_episode": STEPS,
        "control_dt": DT,
        "counts_per_mode": COUNTS_PER_MODE,
        "splits": splits,
    }

    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    # All input validation is complete before publication.
    if OUTPUT.exists():
        raise FileExistsError(
            "Output appeared during validation"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )

    SPLIT_FILE.write_text(
        serialized,
        encoding="utf-8",
    )

    print("SPLIT ASSIGNMENTS SUMMARY")

    for name, records in splits.items():
        counts = Counter(
            item["mode"] for item in records
        )

        print(
            f"SPLIT {name.upper()}"
            f" episodes={len(records)}"
            f" transitions={len(records) * STEPS}"
        )

        print("  MODE COUNTS:", dict(counts))

        print(
            "  EPISODES:",
            " ".join(
                item["file"] for item in records
            ),
        )

    print("SPLIT FILE:", SPLIT_FILE.resolve())
    print("SPLIT FILE SHA256:", sha256(SPLIT_FILE))
    print("M3 MAIN DATASET SPLIT: PASSED")


if __name__ == "__main__":
    main()
