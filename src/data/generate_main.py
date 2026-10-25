"""Generate an expanded, reproducible ActionWorld Lite dataset."""

from collections import Counter
import hashlib
import json
import math
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from src.data.generate_pilot import (
    FIELDS,
    image_64,
    sha256,
    state_vector,
)
from src.simulation.pusht_env import PushTEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/raw/m3_main_v01"

SEED = 20261026
EPISODES = 180
STEPS = 64
IMAGE_SIZE = 64

MODES = (
    "center_push",
    "offset_push",
    "free_motion",
)

EXPECTED_HASHES = {
    "src/simulation/pusht_env.py":
        "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393",
    "src/simulation/pusht_scene.xml":
        "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc",
    "src/data/generate_pilot.py":
        "d2466de34c5303ca86fed608d389d9dc272ada6355f767514e9aaf3321e22eba",
}


def sample_initial(rng, mode):
    block = rng.uniform(-0.10, 0.10, size=2)
    yaw = float(rng.uniform(-math.pi, math.pi))
    theta = float(rng.uniform(-math.pi, math.pi))

    direction = np.array([
        math.cos(theta),
        math.sin(theta),
    ])

    tangent = np.array([
        -direction[1],
        direction[0],
    ])

    if mode == "free_motion":
        signs = rng.choice([-1.0, 1.0], size=2)
        pusher = 0.34 * signs

    else:
        if mode == "center_push":
            offset = float(
                rng.uniform(-0.012, 0.012)
            )
        else:
            offset = float(
                rng.uniform(0.025, 0.055)
                * rng.choice([-1.0, 1.0])
            )

        pusher = (
            block
            - 0.22 * direction
            + offset * tangent
        )

    initial = {
        "block_xy": tuple(float(v) for v in block),
        "block_yaw": yaw,
        "pusher_xy": tuple(float(v) for v in pusher),
    }

    return initial, direction


def choose_action(rng, mode, direction, t):
    if mode == "free_motion":
        delta = rng.uniform(
            -0.003, 0.003, size=2
        )
        return tuple(float(v) for v in delta)

    if t < 30:
        angle = float(rng.uniform(-0.20, 0.20))
        distance = float(rng.uniform(0.012, 0.018))

    elif t < 43:
        angle = math.pi + float(
            rng.uniform(-0.20, 0.20)
        )
        distance = float(rng.uniform(0.006, 0.013))

    else:
        angle = float(rng.uniform(-0.65, 0.65))
        distance = float(rng.uniform(0.008, 0.014))

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated = np.array([
        direction[0] * cos_a - direction[1] * sin_a,
        direction[0] * sin_a + direction[1] * cos_a,
    ])

    delta = distance * rotated

    return tuple(float(v) for v in delta)


def check_source_integrity():
    for filename, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / filename)

        if actual != expected:
            raise RuntimeError(
                f"Source integrity mismatch: {filename}"
            )


def generate():
    check_source_integrity()

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite {OUTPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "schema_version": "m3-main-v0.1",
        "seed": SEED,
        "episodes": EPISODES,
        "steps_per_episode": STEPS,
        "control_dt": 0.05,
        "image_shape": [64, 64, 3],
        "image_downsampling":
            "256x256 nearest-neighbor stride 4",
        "state_fields": list(FIELDS),
        "transition_alignment":
            "states[t], actions[t], states[t+1]; "
            "rgb[t], rgb[t+1]",
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "generator_sha256": sha256(
            Path(__file__).resolve()
        ),
        "source_sha256": EXPECTED_HASHES,
        "modes": list(MODES),
        "splits":
            "not created; complete-episode split required",
        "items": [],
    }

    mode_counts = Counter()
    mode_contacts = Counter()
    moved = Counter()
    rotated = Counter()

    with tempfile.TemporaryDirectory(
        prefix=".m3_main_v01_staging_",
        dir=OUTPUT.parent,
    ) as staging_root:

        staging = Path(staging_root) / "dataset"
        staging.mkdir()

        env = PushTEnv()

        try:
            for i in range(EPISODES):
                mode = MODES[i % len(MODES)]
                episode_seed = SEED + i * 1009

                rng = np.random.default_rng(
                    episode_seed
                )

                kwargs, direction = sample_initial(
                    rng, mode
                )

                first_state = env.reset(**kwargs)

                states = [
                    state_vector(first_state)
                ]
                images = [image_64(env)]
                actions = []

                for t in range(STEPS):
                    action = choose_action(
                        rng, mode, direction, t
                    )

                    next_state = env.step(action)

                    actions.append(action)
                    states.append(
                        state_vector(next_state)
                    )
                    images.append(image_64(env))

                states = np.stack(states)
                actions = np.asarray(
                    actions, dtype=np.float64
                )
                images = np.stack(images)

                assert states.shape == (65, 14)
                assert actions.shape == (64, 2)
                assert images.shape == (65, 64, 64, 3)
                assert images.dtype == np.uint8

                assert np.isfinite(states).all()
                assert np.isfinite(actions).all()

                assert np.allclose(
                    states[:, 0],
                    np.arange(65) * env.control_dt,
                    atol=1e-10,
                    rtol=0,
                )

                assert np.all(
                    np.linalg.norm(actions, axis=1)
                    <= env.max_action_distance + 1e-12
                )

                contacts = int(
                    states[1:, 13].sum()
                )

                net_motion = float(np.linalg.norm(
                    states[-1, 3:5]
                    - states[0, 3:5]
                ))

                yaw_delta = float(
                    states[-1, 6] - states[0, 6]
                )

                net_yaw_deg = math.degrees(
                    math.atan2(
                        math.sin(yaw_delta),
                        math.cos(yaw_delta),
                    )
                )

                filename = f"episode_{i:03d}.npz"
                path = staging / filename

                np.savez_compressed(
                    path,
                    states=states,
                    actions=actions,
                    rgb=images,
                )

                manifest["items"].append({
                    "file": filename,
                    "mode": mode,
                    "episode_seed": episode_seed,
                    "initial": kwargs,
                    "contact_observations": contacts,
                    "block_net_displacement_m": net_motion,
                    "block_net_yaw_deg": net_yaw_deg,
                    "sha256": sha256(path),
                })

                mode_counts[mode] += 1
                mode_contacts[mode] += contacts

                if net_motion > 0.005:
                    moved[mode] += 1

                if abs(net_yaw_deg) > 5.0:
                    rotated[mode] += 1

                if (i + 1) % 15 == 0:
                    print(
                        "GENERATED:",
                        i + 1,
                        "/",
                        EPISODES,
                    )

        finally:
            env.close()

        assert all(
            mode_counts[mode] == 60
            for mode in MODES
        )

        manifest_path = staging / "manifest.json"

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

        if OUTPUT.exists():
            raise FileExistsError(
                "Output appeared during generation"
            )

        # Publish only after the whole dataset succeeds.
        staging.rename(OUTPUT)

    print("MODE SUMMARY")

    for mode in MODES:
        print(
            f"MODE {mode}"
            f" episodes={mode_counts[mode]}"
            f" contacts={mode_contacts[mode]}"
            f" moved_gt_5mm={moved[mode]}"
            f" rotated_gt_5deg={rotated[mode]}"
        )

    print("TOTAL EPISODES:", EPISODES)
    print("TOTAL TRANSITIONS:", EPISODES * STEPS)
    print("TOTAL RGB FRAMES:", EPISODES * (STEPS + 1))
    print("DATASET DIRECTORY:", OUTPUT.resolve())
    print("MANIFEST SHA256:", sha256(
        OUTPUT / "manifest.json"
    ))
    print("M3 MAIN DATASET GENERATION: PASSED")


if __name__ == "__main__":
    generate()
