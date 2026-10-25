"""Read-only physical-data audit for the M3 PushT pilot."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/raw/m3_pilot_v01"

EXPECTED_HASHES = {
    "src/data/generate_pilot.py":
        "d2466de34c5303ca86fed608d389d9dc272ada6355f767514e9aaf3321e22eba",
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
}

MODES = ("center_push", "offset_push", "free_motion")

FIELDS = (
    "time", "pusher_x", "pusher_y",
    "block_x", "block_y", "block_z", "block_yaw",
    "block_qvel_0", "block_qvel_1", "block_qvel_2",
    "block_qvel_3", "block_qvel_4", "block_qvel_5",
    "pusher_contact",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrapped_difference(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def main() -> None:
    for name, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / name)
        assert actual == expected, (
            f"Source hash mismatch: {name}"
        )

    print("PASS: source provenance")

    manifest = json.loads(
        (DATASET / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["schema_version"] == "m3-pilot-v0.1"
    assert manifest["episodes"] == 12
    assert manifest["steps_per_episode"] == 64
    assert manifest["seed"] == 20261025
    assert manifest["state_fields"] == list(FIELDS)
    assert manifest["image_shape"] == [64, 64, 3]
    assert math.isclose(manifest["control_dt"], 0.05)

    assert manifest["generator_sha256"] == (
        EXPECTED_HASHES["src/data/generate_pilot.py"]
    )
    assert manifest["environment_sha256"] == (
        EXPECTED_HASHES["src/simulation/pusht_env.py"]
    )
    assert manifest["scene_sha256"] == (
        EXPECTED_HASHES["src/simulation/pusht_scene.xml"]
    )

    items = manifest["items"]
    assert len(items) == 12

    expected_files = {
        f"episode_{i:03d}.npz"
        for i in range(12)
    }

    actual_files = {
        path.name for path in DATASET.glob("*.npz")
    }

    assert actual_files == expected_files
    assert {item["file"] for item in items} == expected_files

    print("PASS: manifest and file inventory")

    totals = Counter()
    grouped = defaultdict(list)

    print("EPISODE-LEVEL PHYSICAL METRICS")

    for i, item in enumerate(items):
        assert item["file"] == f"episode_{i:03d}.npz"

        path = DATASET / item["file"]
        assert sha256(path) == item["sha256"]

        mode = item["mode"]
        assert mode in MODES

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

            expected_times = np.arange(65) * 0.05
            assert np.allclose(
                states[:, 0],
                expected_times,
                atol=1e-10,
                rtol=0,
            )

            assert np.all(
                np.linalg.norm(actions, axis=1)
                <= 0.020000000001
            )

            # Verify that the recorded pusher positions agree
            # with the actual action sequence.
            expected_pusher = (
                states[0, 1:3]
                + np.vstack((
                    np.zeros((1, 2)),
                    np.cumsum(actions, axis=0),
                ))
            )

            assert np.allclose(
                states[:, 1:3],
                expected_pusher,
                atol=1e-9,
                rtol=0,
            )

            initial = item["initial"]

            assert np.allclose(
                states[0, 3:5],
                initial["block_xy"],
                atol=1e-9,
                rtol=0,
            )
            assert np.allclose(
                states[0, 1:3],
                initial["pusher_xy"],
                atol=1e-9,
                rtol=0,
            )

            initial_yaw_error = math.atan2(
                math.sin(
                    states[0, 6] - initial["block_yaw"]
                ),
                math.cos(
                    states[0, 6] - initial["block_yaw"]
                ),
            )

            assert abs(initial_yaw_error) < 1e-9

            contacts = states[1:, 13]
            assert np.all(
                (contacts == 0) | (contacts == 1)
            )

            contact_count = int(contacts.sum())
            assert contact_count == (
                item["contact_observations"]
            )

            positions = states[:, 3:5]
            position_steps = np.diff(
                positions, axis=0
            )

            block_net = float(np.linalg.norm(
                positions[-1] - positions[0]
            ))

            block_path = float(np.sum(
                np.linalg.norm(
                    position_steps, axis=1
                )
            ))

            yaw_steps = wrapped_difference(
                np.diff(states[:, 6])
            )

            yaw_net = float(np.degrees(
                wrapped_difference(np.array([
                    states[-1, 6] - states[0, 6]
                ]))[0]
            ))

            yaw_path = float(np.degrees(
                np.sum(np.abs(yaw_steps))
            ))

            image_difference = float(np.mean(
                np.abs(
                    rgb[0].astype(np.int16)
                    - rgb[-1].astype(np.int16)
                )
            ))

            assert float(rgb.std()) > 0

            if mode == "free_motion":
                assert contact_count == 0, (
                    "Free motion unexpectedly contacted block"
                )
            else:
                assert contact_count > 0, (
                    "Push episode has no contact"
                )

            metrics = {
                "net_m": block_net,
                "path_m": block_path,
                "net_yaw_deg": yaw_net,
                "yaw_path_deg": yaw_path,
                "contacts": contact_count,
                "image_difference": image_difference,
            }

            grouped[mode].append(metrics)
            totals["contacts"] += contact_count

            if image_difference > 0.05:
                totals["image_changed"] += 1

            if block_net > 0.005:
                totals["block_moved"] += 1

            if abs(yaw_net) > 5.0:
                totals["block_rotated"] += 1

            print(
                f"EPISODE {i:02d}"
                f" mode={mode}"
                f" contact={contact_count}"
                f" block_net_m={block_net:.6f}"
                f" block_path_m={block_path:.6f}"
                f" net_yaw_deg={yaw_net:.3f}"
                f" yaw_path_deg={yaw_path:.3f}"
                f" rgb_diff={image_difference:.4f}"
            )

    assert all(len(grouped[mode]) == 4 for mode in MODES)
    assert totals["contacts"] == 231

    print("MODE-LEVEL SUMMARY")

    for mode in MODES:
        values = grouped[mode]

        mean_net = float(np.mean([
            v["net_m"] for v in values
        ]))

        mean_path = float(np.mean([
            v["path_m"] for v in values
        ]))

        mean_yaw = float(np.mean([
            abs(v["net_yaw_deg"])
            for v in values
        ]))

        mean_yaw_path = float(np.mean([
            v["yaw_path_deg"]
            for v in values
        ]))

        contacts = sum(
            v["contacts"] for v in values
        )

        print(
            f"MODE {mode}"
            f" episodes={len(values)}"
            f" contacts={contacts}"
            f" mean_net_m={mean_net:.6f}"
            f" mean_path_m={mean_path:.6f}"
            f" mean_abs_yaw_deg={mean_yaw:.3f}"
            f" mean_yaw_path_deg={mean_yaw_path:.3f}"
        )

    print("TOTAL CONTACT OBSERVATIONS:", totals["contacts"])
    print("EPISODES WITH IMAGE CHANGE:", totals["image_changed"])
    print("EPISODES WITH BLOCK NET MOVE >5MM:", totals["block_moved"])
    print("EPISODES WITH NET ROTATION >5DEG:", totals["block_rotated"])

    assert totals["image_changed"] == 12

    print("PASS: dataset structure and motion consistency")
    print("PASS: contact versus no-contact mode separation")
    print("PILOT PHYSICAL DATA AUDIT: PASSED")


if __name__ == "__main__":
    main()
