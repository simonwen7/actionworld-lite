# M3 — Dataset Generation and Prediction Baselines

## 1. Research Objective

Build a reproducible ActionWorld Lite PushT-style dataset for
learning action-conditioned physical dynamics.

The physical reference is MuJoCo, not a trained AI model.
AI world-model learning begins in M4.

## 2. Simulation Setup

- Physics engine: MuJoCo 3.17.0
- NumPy: 2.5.4
- Python: 3.12.14
- Physics timestep: 0.005 seconds
- Control interval: 0.05 seconds
- Physics steps per action: 10
- Action: pusher displacement (dx, dy)
- Maximum action distance: 0.02 meters
- Scene: simplified MuJoCo PushT-style manipulation
- RGB data: 64 x 64 x 3, uint8
- Image acquisition: 256 x 256 top-camera rendering, then
  nearest-neighbor stride-4 subsampling

This environment is not an official PushT benchmark reproduction.

The pusher is kinematically controlled through a MuJoCo
mocap body. The T-shaped block responds through physical
contact simulation.

## 3. Dataset Schema

Each episode contains:

- states: shape (65, 14), float64
- actions: shape (64, 2), float64
- rgb: shape (65, 64, 64, 3), uint8

The fourteen state fields are, in order:

1. time
2. pusher_x
3. pusher_y
4. block_x
5. block_y
6. block_z
7. block_yaw
8. block_qvel_0
9. block_qvel_1
10. block_qvel_2
11. block_qvel_3
12. block_qvel_4
13. block_qvel_5
14. pusher_contact

Transition t is:

states[t], actions[t] -> states[t+1]

The associated image pair is:

rgb[t], rgb[t+1]

The contact flag is an observed simulator label.
A label from the next state must never be used as an input
when predicting that next state.

## 4. Dataset Generation

Pilot:
- 12 episodes
- 768 transitions
- 780 RGB frames
- Seed: 20261025
- Purpose: pipeline validation

Main:
- 180 episodes
- 11,520 transitions
- 11,700 RGB frames
- Seed: 20261026
- Each episode contains 64 actions
- Per-episode RNG seed: 20261026 + episode_index * 1009

Main dataset mode allocation:

- center_push: 60 episodes
- offset_push: 60 episodes
- free_motion: 60 episodes

The generator randomizes initial block pose, pusher
position, push direction, offset and action magnitudes
within the defined sampling policy.

The dataset does not randomize physical parameters such
as mass, friction, geometry or gravity.

Main observed physical coverage:

| Mode | Contact observations | Moved >5 mm | Rotated >5 deg |
|---|---:|---:|---:|
| center_push | 1859 | 60/60 | 57/60 |
| offset_push | 1337 | 60/60 | 51/60 |
| free_motion | 0 | 0/60 | 0/60 |

Contact observations are sampled at action endpoints.
They do not represent every MuJoCo physics substep.

## 5. Data Splits

Main split policy:

- Train: 126 episodes, 8064 transitions
- Validation: 27 episodes, 1728 transitions
- Test: 27 episodes, 1728 transitions

Every mode is represented as follows:

- Train: 42 episodes per mode
- Validation: 9 episodes per mode
- Test: 9 episodes per mode

Splits are assigned by complete episode.

The split implementation uses deterministic SHA-256 ranking
within each mode.

No episode is shared across train, validation or test.

The split is frozen for subsequent evaluation unless a new
dataset version is explicitly defined.

## 6. Prediction Baselines

Both baselines predict the next T-block XY position and
planar yaw at a 0.05-second horizon.

Persistence:
- Next XY equals current XY
- Next yaw equals current yaw

Constant Velocity:
- Next XY = current XY + dt * current XY velocity
- Next yaw = current yaw + dt * current z angular velocity

The yaw extrapolation is a planar approximation.

Constant Velocity uses simulator-provided velocity.
Therefore it is a privileged-state baseline.

Neither baseline uses future actions or future states
as prediction inputs.

The actions are recorded for future action-conditioned
world-model training, but these baselines are not
action-conditioned.

## 7. Main Dataset Results

Position metric: mean Euclidean XY error in centimeters.
Yaw metric: mean absolute shortest-angle error in degrees.

| Split | Baseline | Position cm | Yaw deg |
|---|---|---:|---:|
| Train | Persistence | 0.241582 | 0.676171 |
| Train | Constant Velocity | 0.042397 | 0.201088 |
| Validation | Persistence | 0.241708 | 0.593354 |
| Validation | Constant Velocity | 0.040910 | 0.187613 |
| Test | Persistence | 0.250882 | 0.694828 |
| Test | Constant Velocity | 0.039333 | 0.183736 |

Test contact subset:

- Contact transitions: 464
- Persistence position error: 0.915546 cm
- Constant Velocity position error: 0.128964 cm
- Persistence yaw error: 2.540684 deg
- Constant Velocity yaw error: 0.631303 deg

Test no-contact subset:

- No-contact transitions: 1264
- Persistence position error: 0.006891 cm
- Constant Velocity position error: 0.006431 cm
- Persistence yaw error: 0.017235 deg
- Constant Velocity yaw error: 0.019439 deg

Contact grouping is computed after prediction using the
target state's contact label. It is never a predictor input.

## 8. Provenance

Reference source SHA-256 values:

- MuJoCo scene:
  a2aa4976dd8757784e4004f2a8652b822ec53b7aff74e880ed675d2fdaa678bc

- PushT environment:
  99cb0a9b37c2ab6516970e9390ae1d509f5db6acd1f4c5efe92bbc810c5e1393

- Main generator:
  4588785fd5deec05d877d4154352b6c910ecd91cb11860ccdc5d67e3305de8d8

- Main split implementation:
  91b0188b7ab7a9eb2a9842b5d212967f654316eb1528776ecf17ad373332d1fc

- Main baseline evaluator:
  2b49452f9070642f65adf9cf9bf2c9779ecee59338c992a40f4fbd5b1196574e

Reference generated artifact SHA-256 values:

- Main manifest:
  2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50

- Main splits:
  af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa

- Main baseline results:
  a545877087a07471a194d942392ddf1a9ec612464bbfa0179280e7554c68eac8

These hashes describe the reference local run.
Bitwise identity across different hardware, graphics drivers,
platforms or dependency environments is not guaranteed.

## 9. Rebuilding the Data

From a fresh checkout with Python 3.12 and uv installed:

uv venv --python python3.12 .venv

uv pip install --python .venv/bin/python -r requirements-m3.txt

Run from the repository root:

.venv/bin/python -B -m src.data.generate_pilot

.venv/bin/python -B -m src.data.audit_pilot

.venv/bin/python -B -m src.data.split_pilot

.venv/bin/python -B -m src.data.evaluate_baselines

.venv/bin/python -B -m src.data.generate_main

.venv/bin/python -B -m src.data.split_main

.venv/bin/python -B -m src.data.evaluate_main_baselines

Generation requires MuJoCo offscreen rendering and a usable
graphics context. On macOS, a normal graphical user session may
be required.

The scripts refuse to overwrite existing dataset outputs.
These reproduction commands are intended for a fresh checkout
without previously generated data.

Reference dataset paths:

data/raw/m3_pilot_v01/
data/raw/m3_main_v01/
data/processed/m3_pilot_v01/
data/processed/m3_main_v01/

Generated data and result JSON files are intentionally Git-ignored.
GitHub contains the generators, evaluators and research protocol,
not the full dataset archives.

The requirements file pins direct dependencies but does not
constitute a full transitive dependency lockfile.

## 10. Limitations

1. This is a simplified PushT-style simulation, not a
   real-world robotics dataset.
2. The pusher follows kinematic actions, rather than
   learned robot control or full actuator dynamics.
3. The dataset has three deliberately constructed
   action-policy modes.
4. Physical parameters are fixed across episodes.
5. Images are downsampled by stride-based subsampling.
6. The test split measures held-out episodes from the same
   simulation family, not unseen physical domains.
7. The reported baselines predict one step, not long rollouts.
8. The constant-velocity baseline receives privileged
   simulator velocity information.
9. A future RGB-only model will have different input access.
10. No trained AI world model exists at the end of M3.

## 11. M4 Handoff

M4 will develop an actually trained action-conditioned
world model using the generated dataset.

Training inputs, target representation and model architecture
must be explicitly decided before M4 implementation.

Evaluation must preserve the existing train/validation/test
episode split and distinguish:

- one-step prediction
- multi-step rollout drift
- contact versus non-contact performance
- state prediction versus visual prediction
- privileged-state versus visual-input comparisons

The test set should not be used for training or iterative
model selection.

End of M3 research record.
