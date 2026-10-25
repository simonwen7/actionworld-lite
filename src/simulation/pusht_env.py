"""Deterministic MuJoCo controller for the ActionWorld Lite PushT-style scene."""

from pathlib import Path
import math

import mujoco
import numpy as np


class PushTEnv:
    """A simple action-controlled tabletop pushing environment.

    Actions are (dx, dy) displacements in meters per control step.
    The pusher is kinematic; the T-block is simulated dynamically.
    This is a physics reference environment, not a learned world model.
    """

    def __init__(
        self,
        scene_path: str | Path | None = None,
        physics_steps_per_action: int = 10,
        max_action_distance: float = 0.02,
    ) -> None:
        if (
            isinstance(physics_steps_per_action, bool)
            or not isinstance(physics_steps_per_action, int)
            or physics_steps_per_action <= 0
        ):
            raise ValueError("physics_steps_per_action must be a positive integer")

        if not math.isfinite(max_action_distance) or max_action_distance <= 0:
            raise ValueError("max_action_distance must be positive and finite")

        if scene_path is None:
            scene_path = Path(__file__).resolve().with_name("pusht_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.physics_steps_per_action = physics_steps_per_action
        self.max_action_distance = float(max_action_distance)

        self._pusher_body_id = self.model.body("pusher").id
        self._block_body_id = self.model.body("t_block").id

        self._mocap_id = int(
            self.model.body_mocapid[self._pusher_body_id]
        )
        if self._mocap_id < 0:
            raise ValueError("The pusher must be a mocap body")

        self._initial_pusher_pos = self.model.body_pos[
            self._pusher_body_id
        ].copy()

        self._pusher_geom_id = self.model.geom("pusher_geom").id
        self._block_geom_ids = {
            self.model.geom("t_bar").id,
            self.model.geom("t_stem").id,
        }

        joint_id = self.model.joint("t_free").id
        self._block_dof_address = int(self.model.jnt_dofadr[joint_id])
        self._block_qpos_address = int(self.model.jnt_qposadr[joint_id])

        self._renderer: mujoco.Renderer | None = None
        self.reset()

    @property
    def control_dt(self) -> float:
        """Simulated seconds advanced by one action."""
        return (
            self.physics_steps_per_action
            * float(self.model.opt.timestep)
        )

    def reset(
        self,
        *,
        block_xy: tuple[float, float] | None = None,
        block_yaw: float = 0.0,
        pusher_xy: tuple[float, float] | None = None,
    ) -> dict[str, object]:
        """Reset to a valid, explicitly configured initial state.

        No arguments preserve the original M2 reset behavior.
        All inputs are validated before the physics state is changed.
        """
        block = (
            np.zeros(2, dtype=np.float64)
            if block_xy is None
            else np.asarray(block_xy, dtype=np.float64)
        )

        pusher = (
            self._initial_pusher_pos[:2].copy()
            if pusher_xy is None
            else np.asarray(pusher_xy, dtype=np.float64)
        )

        if block.shape != (2,) or not np.all(np.isfinite(block)):
            raise ValueError("block_xy must contain two finite numbers")

        if pusher.shape != (2,) or not np.all(np.isfinite(pusher)):
            raise ValueError("pusher_xy must contain two finite numbers")

        try:
            yaw = float(block_yaw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("block_yaw must be finite") from exc

        if not math.isfinite(yaw) or abs(yaw) > math.pi:
            raise ValueError("block_yaw must be within [-pi, pi]")

        if np.any(np.abs(block) > 0.25):
            raise ValueError("block_xy exceeds the initial-state bounds")

        if np.any(np.abs(pusher) > 0.45):
            raise ValueError("pusher_xy exceeds the initial-state bounds")

        if np.linalg.norm(pusher - block) < 0.16:
            raise ValueError("Initial pusher and T-block are too close")

        mujoco.mj_resetData(self.model, self.data)

        address = self._block_qpos_address

        self.data.qpos[address : address + 2] = block

        self.data.qpos[address + 3 : address + 7] = [
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ]

        self.data.mocap_pos[self._mocap_id, :2] = pusher

        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, action: tuple[float, float]) -> dict[str, object]:
        """Move the pusher incrementally and advance the simulation.

        Each action represents a total XY displacement in meters.
        The displacement is divided evenly across physics substeps.
        """
        delta = np.asarray(action, dtype=np.float64)

        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("action must contain two finite numbers")

        if np.linalg.norm(delta) > self.max_action_distance + 1e-12:
            raise ValueError(
                "action exceeds max_action_distance"
            )

        initial_pos = self.data.mocap_pos[
            self._mocap_id
        ].copy()

        for i in range(self.physics_steps_per_action):
            fraction = (i + 1) / self.physics_steps_per_action

            self.data.mocap_pos[self._mocap_id, :2] = (
                initial_pos[:2] + fraction * delta
            )

            mujoco.mj_step(self.model, self.data)

        return self.observe()

    def _has_pusher_contact(self) -> bool:
        """Check whether the pusher touches either T-block geometry."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}

            if (
                self._pusher_geom_id in pair
                and bool(pair & self._block_geom_ids)
            ):
                return True

        return False

    def observe(self) -> dict[str, object]:
        """Return an independent snapshot of the current state."""
        block_pos = self.data.xpos[self._block_body_id]
        rotation = self.data.xmat[
            self._block_body_id
        ].reshape(3, 3)

        yaw = math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0]),
        )

        velocity_start = self._block_dof_address
        block_velocity = self.data.qvel[
            velocity_start : velocity_start + 6
        ]

        return {
            "time": float(self.data.time),
            "pusher_xy": tuple(
                float(v)
                for v in self.data.mocap_pos[self._mocap_id, :2]
            ),
            "block_xy": tuple(
                float(v) for v in block_pos[:2]
            ),
            "block_z": float(block_pos[2]),
            "block_yaw": yaw,
            "block_linear_velocity": tuple(
                float(v) for v in block_velocity[:3]
            ),
            "block_angular_velocity": tuple(
                float(v) for v in block_velocity[3:]
            ),
            "pusher_contact": self._has_pusher_contact(),
        }

    def render(self) -> np.ndarray:
        """Return a 256x256 RGB image from the top camera."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=256,
                width=256,
            )

        self._renderer.update_scene(self.data, camera="top")
        return self._renderer.render().copy()

    def close(self) -> None:
        """Release the offscreen rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
