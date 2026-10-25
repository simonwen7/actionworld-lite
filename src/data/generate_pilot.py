"""Produce a small reproducible PushT pilot dataset."""

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from src.simulation.pusht_env import PushTEnv


SEED = 20261025
EPISODES = 12
STEPS = 64
IMAGE_SIZE = 64
OUTPUT = Path("data/raw/m3_pilot_v01")

MODES = ("center_push", "offset_push", "free_motion")

FIELDS = (
    "time", "pusher_x", "pusher_y", "block_x", "block_y",
    "block_z", "block_yaw", "block_qvel_0", "block_qvel_1",
    "block_qvel_2", "block_qvel_3", "block_qvel_4",
    "block_qvel_5", "pusher_contact",
)

EXPECTED_ENV_SHA256 = (
    "99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393"
)
EXPECTED_SCENE_SHA256 = (
    "a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_vector(obs: dict[str, object]) -> np.ndarray:
    return np.array([
        obs["time"],
        *obs["pusher_xy"],
        *obs["block_xy"],
        obs["block_z"],
        obs["block_yaw"],
        *obs["block_linear_velocity"],
        *obs["block_angular_velocity"],
        float(obs["pusher_contact"]),
    ], dtype=np.float64)


def sample_initial(rng: np.random.Generator, mode: str):
    block = rng.uniform(-0.08, 0.08, size=2)
    yaw = float(rng.uniform(-math.pi, math.pi))
    theta = float(rng.uniform(-math.pi, math.pi))

    direction = np.array([
        math.cos(theta),
        math.sin(theta),
    ])
    tangent = np.array([-direction[1], direction[0]])

    if mode == "free_motion":
        signs = rng.choice([-1.0, 1.0], size=2)
        pusher = 0.34 * signs
    else:
        offset = (
            0.0
            if mode == "center_push"
            else 0.035 * float(rng.choice([-1.0, 1.0]))
        )
        pusher = (
            block - 0.22 * direction + offset * tangent
        )

    return {
        "block_xy": tuple(float(x) for x in block),
        "block_yaw": yaw,
        "pusher_xy": tuple(float(x) for x in pusher),
    }, direction


def choose_action(
    rng: np.random.Generator,
    mode: str,
    direction: np.ndarray,
    t: int,
):
    if mode == "free_motion":
        return tuple(
            float(x)
            for x in rng.uniform(-0.003, 0.003, size=2)
        )

    if t < 32:
        delta = 0.015 * direction
    elif t < 48:
        delta = -0.012 * direction
    else:
        sign = 1.0 if mode == "offset_push" else -1.0
        angle = 0.45 * sign

        rotated = np.array([
            direction[0] * math.cos(angle)
            - direction[1] * math.sin(angle),
            direction[0] * math.sin(angle)
            + direction[1] * math.cos(angle),
        ])
        delta = 0.012 * rotated

    return tuple(float(x) for x in delta)


def image_64(env: PushTEnv) -> np.ndarray:
    frame = env.render()

    if (
        frame.shape != (256, 256, 3)
        or frame.dtype != np.uint8
    ):
        raise RuntimeError("Unexpected renderer output")

    return frame[::4, ::4].copy()


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    env_path = repo / "src/simulation/pusht_env.py"
    scene_path = repo / "src/simulation/pusht_scene.xml"

    if sha256(env_path) != EXPECTED_ENV_SHA256:
        raise RuntimeError(
            "Unexpected PushTEnv source version; stop"
        )

    if sha256(scene_path) != EXPECTED_SCENE_SHA256:
        raise RuntimeError(
            "Unexpected MuJoCo scene version; stop"
        )

    output = repo / OUTPUT

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}"
        )

    rng = np.random.default_rng(SEED)
    records = []
    env = PushTEnv()

    try:
        for i in range(EPISODES):
            mode = MODES[i % len(MODES)]

            initial_kwargs, direction = sample_initial(
                rng, mode
            )

            first_state = env.reset(**initial_kwargs)

            states = [state_vector(first_state)]
            images = [image_64(env)]
            actions = []

            for t in range(STEPS):
                action = choose_action(
                    rng, mode, direction, t
                )

                next_state = env.step(action)

                actions.append(action)
                states.append(state_vector(next_state))
                images.append(image_64(env))

            states = np.stack(states)
            actions = np.asarray(
                actions, dtype=np.float64
            )
            images = np.stack(images)

            if states.shape != (
                STEPS + 1, len(FIELDS)
            ):
                raise RuntimeError("Incorrect state shape")

            if actions.shape != (STEPS, 2):
                raise RuntimeError("Incorrect action shape")

            if images.shape != (
                STEPS + 1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            ):
                raise RuntimeError("Incorrect image shape")

            if (
                not np.isfinite(states).all()
                or not np.isfinite(actions).all()
            ):
                raise RuntimeError("Non-finite data")

            if np.max(
                np.linalg.norm(actions, axis=1)
            ) > env.max_action_distance + 1e-12:
                raise RuntimeError("Invalid action")

            expected_times = (
                np.arange(STEPS + 1) * env.control_dt
            )

            if not np.allclose(
                states[:, 0],
                expected_times,
                atol=1e-10,
            ):
                raise RuntimeError("Incorrect time alignment")

            contact_steps = int(states[1:, -1].sum())

            records.append((
                i,
                mode,
                initial_kwargs,
                states,
                actions,
                images,
                contact_steps,
            ))

            print(
                f"EPISODE {i:02d}"
                f" mode={mode}"
                f" contacts={contact_steps}"
                f" images={len(images)}"
            )

    finally:
        env.close()

    # Only write after in-memory generation succeeds.
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}"
        )

    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": "m3-pilot-v0.1",
        "seed": SEED,
        "episodes": EPISODES,
        "steps_per_episode": STEPS,
        "control_dt": 0.05,
        "image_shape": [
            IMAGE_SIZE, IMAGE_SIZE, 3
        ],
        "image_downsampling":
            "256x256 nearest-neighbor stride 4",
        "state_fields": list(FIELDS),
        "transition_alignment":
            "states[t], actions[t], states[t+1]; "
            "rgb[t], rgb[t+1]",
        "mujoco_version": mujoco.__version__,
        "environment_sha256": sha256(env_path),
        "scene_sha256": sha256(scene_path),
        "generator_sha256": sha256(Path(__file__)),
        "splits":
            "not created; episode-level split required later",
        "items": [],
    }

    for (
        i, mode, kwargs, states, actions,
        images, contact_steps
    ) in records:
        name = f"episode_{i:03d}.npz"

        np.savez_compressed(
            output / name,
            states=states,
            actions=actions,
            rgb=images,
        )

        manifest["items"].append({
            "file": name,
            "mode": mode,
            "initial": kwargs,
            "contact_observations": contact_steps,
            "sha256": sha256(output / name),
        })

    (output / "manifest.json").write_text(
        json.dumps(
            manifest, indent=2, sort_keys=True
        ) + "\n",
        encoding="utf-8",
    )

    print("PILOT OUTPUT:", output.resolve())
    print("TRANSITIONS:", EPISODES * STEPS)
    print(
        "RGB FRAMES:",
        EPISODES * (STEPS + 1),
    )
    print(
        "CONTACT OBSERVATIONS:",
        sum(r[-1] for r in records),
    )
    print("M3 PILOT GENERATION: PASSED")


if __name__ == "__main__":
    main()
