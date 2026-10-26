"""M5-04: macOS-local, real-observation-conditioned PushT one-step demo.

This is NOT autonomous free-running simulation or image prediction.
The existing M4 model and consumed held-out TEST are read-only and untouched.

Task M5-04 authorizes only ``--self-test`` (no GUI/model/simulation).
A later, separate approval is required to execute ``--run-demo``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 256
MODEL_IMAGE_SIZE = 64
ACTION_SCALE_M = 0.02
CANVAS_SIZE = 330
VIEW_HALF_EXTENT_M = 0.55


# Pure helpers: imported and self-tested without touching the simulator or model.
def frame_to_model_rgb(frame: np.ndarray) -> np.ndarray:
    """M3 image_64 equivalent: original 256 RGB -> origin stride-4 uint8."""
    if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
        raise ValueError("MuJoCo render must be 256x256x3 uint8 RGB")
    result = frame[::4, ::4].copy()
    if result.shape != (64, 64, 3):
        raise AssertionError("Unexpected stride-4 RGB output")
    return result


def make_model_inputs(previous: np.ndarray, current: np.ndarray,
                      action_m: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 arrays [1,6,64,64] and [1,2]; no privileged state."""
    for frame in (previous, current):
        if frame.shape != (64, 64, 3) or frame.dtype != np.uint8:
            raise ValueError("History frames must be 64x64x3 uint8 RGB")
    displacement = np.asarray(action_m, dtype=np.float64)
    if displacement.shape != (2,) or not np.isfinite(displacement).all():
        raise ValueError("Action must contain two finite meter displacements")
    if float(np.linalg.norm(displacement)) > ACTION_SCALE_M + 1e-12:
        raise ValueError("Action norm cannot exceed 0.02 m")
    channels = np.concatenate((previous.transpose(2, 0, 1),
                               current.transpose(2, 0, 1)), axis=0)
    frames = np.ascontiguousarray(channels[None].astype(np.float32) / 255.0)
    action = np.ascontiguousarray((displacement / ACTION_SCALE_M)[None],
                                 dtype=np.float32)
    return frames, action


def ppm_bytes(rgb: np.ndarray) -> bytes:
    """Tk's built-in PPM decoder; no Pillow, disk temp file, or dependency."""
    if (rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3)
            or rgb.dtype != np.uint8):
        raise ValueError("Display frame must be 256x256x3 uint8 RGB")
    return (f"P6\n{IMAGE_SIZE} {IMAGE_SIZE}\n255\n".encode("ascii")
            + rgb.tobytes(order="C"))


def yaw_of(cs: tuple[float, float]) -> float:
    return math.atan2(float(cs[1]), float(cs[0]))


def circular_yaw_error_deg(pred: float, truth: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(pred - truth),
                                       math.cos(pred - truth))))


def world_to_canvas(x: float, y: float) -> tuple[float, float]:
    scale = CANVAS_SIZE / (2.0 * VIEW_HALF_EXTENT_M)
    return CANVAS_SIZE / 2 + x * scale, CANVAS_SIZE / 2 - y * scale


def box_outline(block_xy: tuple[float, float], yaw: float,
                center_y: float, half_x: float, half_y: float) -> list[float]:
    """A world-coordinate 2D T-box polygon. Never changes MuJoCo MjData."""
    c, s = math.cos(yaw), math.sin(yaw)
    result: list[float] = []
    for lx, ly in ((-half_x, center_y - half_y),
                   (half_x, center_y - half_y),
                   (half_x, center_y + half_y),
                   (-half_x, center_y + half_y)):
        x = block_xy[0] + c * lx - s * ly
        y = block_xy[1] + s * lx + c * ly
        px, py = world_to_canvas(x, y)
        result.extend((px, py))
    return result


def self_test() -> None:
    """Synthetic-only QA. No datasets, MuJoCo calls, or model imports."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256))
    img[..., 0] = yy.astype(np.uint8)
    img[..., 1] = xx.astype(np.uint8)
    img[..., 2] = 127
    small = frame_to_model_rgb(img)
    assert small.shape == (64, 64, 3)
    assert np.array_equal(small, img[::4, ::4])
    assert int(small[3, 5, 0]) == 12
    assert int(small[3, 5, 1]) == 20
    old = np.zeros_like(small)
    old[..., 1] = 255
    frames, action = make_model_inputs(old, small, (0.012, -0.016))
    assert frames.shape == (1, 6, 64, 64) and frames.dtype == np.float32
    assert action.shape == (1, 2) and action.dtype == np.float32
    assert np.allclose(action, [[0.6, -0.8]], atol=1e-6)
    assert frames[0, 1, 0, 0] == 1.0
    assert np.isclose(frames[0, 3, 3, 5], small[3, 5, 0] / 255.0, atol=1e-7)
    assert circular_yaw_error_deg(math.radians(179), math.radians(-179)) < 2.001
    assert abs(yaw_of((0, 1)) - math.pi / 2) < 1e-12
    assert world_to_canvas(0, 0) == (CANVAS_SIZE / 2, CANVAS_SIZE / 2)
    assert len(box_outline((0.0, 0.0), 0.0, 0.045, 0.095, 0.025)) == 8
    assert len(box_outline((0.0, 0.0), 0.0, -0.020, 0.025, 0.060)) == 8
    assert ppm_bytes(img).startswith(b"P6\n256 256\n255\n")
    try:
        make_model_inputs(old, small, (0.02, 0.02))
    except ValueError:
        pass
    else:
        raise AssertionError("Exceeded action norm was accepted")
    try:
        frame_to_model_rgb(np.zeros((64, 64, 3), dtype=np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong render shape was accepted")
    print("M5-04 SYNTHETIC-ONLY SELF-TEST: PASSED")
    print("NO MuJoCo, model inference, GUI, TRAIN, VALIDATION, or TEST opened")


class InteractiveDemo:
    """One-step predictions conditioned on TWO REAL observed frames."""

    def __init__(self, root: tk.Tk, env, model, torch_module):
        self.root = root
        self.env = env
        self.model = model
        self.torch = torch_module
        self.previous: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.observation: dict[str, object] | None = None
        self.last_pred: dict[str, object] | None = None
        self._photo: tk.PhotoImage | None = None
        self.step_count = 0
        self.busy = False
        self.failed = False

        root.title("ActionWorld Lite | M5 PushT one-step model comparison")
        root.geometry("1000x715")
        root.minsize(920, 650)
        outer = ttk.Frame(root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="ActionWorld Lite — Interactive one-step comparison",
                  font=("Helvetica", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text=("Orange = real MuJoCo block; blue dashed = predicted next pose. "
                               "Prediction is state geometry, not a generated image."),
                  wraplength=940).pack(anchor="w", pady=(3, 12))
        content = ttk.Frame(outer)
        content.pack(fill=tk.X)
        left = ttk.LabelFrame(content, text="Real MuJoCo RGB (256×256)", padding=8)
        left.grid(row=0, column=0, sticky="n", padx=(0, 12))
        self.real_image = ttk.Label(left)
        self.real_image.pack()
        ttk.Label(left, text="Live 256×256 rendering; model uses origin stride-4 64×64",
                  wraplength=280).pack(pady=(8, 0))

        right = ttk.LabelFrame(content, text="World coordinates — XY / meters", padding=8)
        right.grid(row=0, column=1, sticky="n", padx=(0, 12))
        self.canvas = tk.Canvas(right, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#F8FAFC", highlightthickness=0)
        self.canvas.pack()
        ttk.Label(right, text="Orange = actual next; dashed blue = predicted next",
                  wraplength=340).pack(pady=(8, 0))

        info = ttk.LabelFrame(content, text="One-step measurements", padding=12)
        info.grid(row=0, column=2, sticky="nsew")
        content.columnconfigure(2, weight=1)
        self.measurements = tk.StringVar(value="No action evaluated yet.")
        ttk.Label(info, textvariable=self.measurements, justify=tk.LEFT,
                  wraplength=230, font=("Menlo", 10)).pack(anchor="nw")

        controls = ttk.LabelFrame(outer, text="Actions (simulated XY displacement)",
                                  padding=10)
        controls.pack(fill=tk.X, pady=(15, 8))
        self.mm = tk.StringVar(value="20")
        ttk.Label(controls, text="Step size:").grid(row=0, column=0, padx=5)
        for j, value in enumerate(("5", "10", "20"), 1):
            ttk.Radiobutton(controls, text=f"{value} mm", variable=self.mm,
                            value=value).grid(row=0, column=j, padx=4)
        self.buttons: list[ttk.Button] = []
        for col, title, direction in ((1, "← / A", (-1, 0)),
                                       (2, "↑ / W", (0, 1)),
                                       (3, "↓ / S", (0, -1)),
                                       (4, "→ / D", (1, 0)),
                                       (5, "Zero", (0, 0))):
            button = ttk.Button(controls, text=title,
                                command=lambda vec=direction: self.act(vec))
            button.grid(row=1, column=col, padx=5, pady=7, sticky="ew")
            self.buttons.append(button)
        self.reset_button = ttk.Button(controls, text="Reset", command=self.reset)
        self.reset_button.grid(row=1, column=0, padx=5, pady=7)
        ttk.Label(controls, text="Arrow keys or WASD. Every press advances one real 0.05 s step.",
                  wraplength=900).grid(row=2, column=0, columnspan=6, sticky="w")
        for keysym, vector in (("Left", (-1, 0)), ("Right", (1, 0)),
                               ("Up", (0, 1)), ("Down", (0, -1)),
                               ("a", (-1, 0)), ("d", (1, 0)),
                               ("w", (0, 1)), ("s", (0, -1))):
            root.bind(f"<KeyPress-{keysym}>",
                      lambda _event, vec=vector: self.act(vec))

        self.status = tk.StringVar(value="Starting...")
        ttk.Label(outer, textvariable=self.status, wraplength=950).pack(anchor="w", pady=6)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.reset()

    def _draw_real_frame(self, rgb: np.ndarray) -> None:
        self._photo = tk.PhotoImage(data=ppm_bytes(rgb), format="PPM")
        self.real_image.configure(image=self._photo)

    def _draw_box(self, xy: tuple[float, float], yaw: float,
                  *, outline: str, fill: str = "", dashed: bool = False) -> None:
        extra = {"dash": (5, 3)} if dashed else {}
        for center_y, half_x, half_y in ((0.045, 0.095, 0.025),
                                          (-0.020, 0.025, 0.060)):
            self.canvas.create_polygon(box_outline(xy, yaw, center_y, half_x, half_y),
                                       outline=outline, fill=fill, width=2, **extra)

    def _draw_world(self) -> None:
        self.canvas.delete("all")
        center = CANVAS_SIZE // 2
        for i in range(0, CANVAS_SIZE + 1, 30):
            self.canvas.create_line(i, 0, i, CANVAS_SIZE, fill="#E2E8F0")
            self.canvas.create_line(0, i, CANVAS_SIZE, i, fill="#E2E8F0")
        self.canvas.create_line(center, 0, center, CANVAS_SIZE, fill="#94A3B8")
        self.canvas.create_line(0, center, CANVAS_SIZE, center, fill="#94A3B8")
        if self.observation is None:
            return
        observed = self.observation
        block = tuple(observed["block_xy"])
        self._draw_box(block, float(observed["block_yaw"]),
                       outline="#A94F20", fill="#F59E5B")
        px, py = world_to_canvas(*observed["pusher_xy"])
        scale = CANVAS_SIZE / (2 * VIEW_HALF_EXTENT_M)
        r = 0.025 * scale
        self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                fill="#3B82C4", outline="#194C84", width=2)
        if self.last_pred is not None:
            self._draw_box(self.last_pred["next_xy"], self.last_pred["next_yaw"],
                           outline="#0369A1", dashed=True)
        self.canvas.create_text(9, 9, text="+Y ↑       +X →", fill="#334155",
                                anchor="nw", font=("Helvetica", 10))
        self.canvas.create_text(9, CANVAS_SIZE - 9,
                                text="World XY, NOT pixel registration",
                                fill="#64748B", anchor="sw", font=("Helvetica", 9))

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy or self.failed else tk.NORMAL
        for button in self.buttons:
            button.configure(state=state)
        self.reset_button.configure(state=tk.DISABLED if busy else tk.NORMAL)

    def reset(self) -> None:
        if self.busy:
            return
        self._set_busy(True)
        try:
            self.env.reset()  # fresh simulator state; no stored dataset
            first_high = self.env.render()
            prev = frame_to_model_rgb(first_high)
            self.env.step((0.0, 0.0))  # real initialization step, not scored
            self.observation = self.env.observe()
            second_high = self.env.render()
            curr = frame_to_model_rgb(second_high)
            self.previous, self.current = prev, curr
            self.last_pred = None
            self.step_count = 0
            self.failed = False
            self._draw_real_frame(second_high)
            self._draw_world()
            self.measurements.set("Ready.\n0 scored actions.\nStartup zero-action step excluded.")
            self.status.set("Ready: choose a direction. Live state initialized from two real frames.")
        except Exception as exc:
            self.failed = True
            self.status.set(f"RESET FAILED — {type(exc).__name__}: {exc}")
        finally:
            self._set_busy(False)

    def act(self, direction: tuple[int, int]) -> None:
        if self.busy or self.failed or self.current is None or self.previous is None:
            return
        try:
            magnitude = int(self.mm.get()) / 1000.0
            if magnitude not in (0.005, 0.01, 0.02):
                raise ValueError("Unsupported action step size")
            action_m = (direction[0] * magnitude, direction[1] * magnitude)
            frame_data, action_data = make_model_inputs(self.previous, self.current,
                                                        action_m)
        except Exception as exc:
            self.status.set(f"INVALID ACTION — {type(exc).__name__}: {exc}")
            return
        self._set_busy(True)
        try:
            before = self.observation
            if before is None:
                raise RuntimeError("Missing prior physical observation")
            # Imports at --run-demo startup, inference exclusively CPU and before step.
            frames = self.torch.from_numpy(frame_data)
            action = self.torch.from_numpy(action_data)
            with self.torch.inference_mode():
                result = self.model(frames, action)
            xy = tuple(float(v) for v in result["next_xy"][0].tolist())
            cs = tuple(float(v) for v in result["next_yaw_cs"][0].tolist())
            delta = tuple(float(v) for v in result["delta_xy"][0].tolist())
            pred = {
                "next_xy": xy,
                "next_yaw": yaw_of(cs),
                "delta_xy": delta,
                "contact": bool(result["next_contact"][0].item()),
                "score_m": float(result["contact_score_m"][0].item()),
            }
            # Absolutely no second inference. Match EXACT same physical action.
            after = self.env.step(action_m)
            displayed_high = self.env.render()
            following = frame_to_model_rgb(displayed_high)
            true_next_xy = tuple(after["block_xy"])
            xy_err_cm = 100.0 * math.dist(pred["next_xy"], true_next_xy)
            yaw_err_deg = circular_yaw_error_deg(pred["next_yaw"],
                                                 float(after["block_yaw"]))
            actual_delta = (true_next_xy[0] - before["block_xy"][0],
                            true_next_xy[1] - before["block_xy"][1])
            motion_err_cm = 100.0 * math.dist(pred["delta_xy"], actual_delta)
            contact_actual = bool(after["pusher_contact"])
            self.previous, self.current = self.current, following
            self.observation = after
            self.last_pred = pred
            self.step_count += 1
            self._draw_real_frame(displayed_high)
            self._draw_world()
            self.measurements.set(
                f"Scored actions: {self.step_count}\n"
                f"Sim time: {after['time']:.3f} s\n\n"
                f"Next XY error: {xy_err_cm:.3f} cm\n"
                f"Next yaw error: {yaw_err_deg:.3f}°\n"
                f"Motion error: {motion_err_cm:.3f} cm\n\n"
                f"Pred XY: ({xy[0]:+.3f}, {xy[1]:+.3f}) m\n"
                f"Real XY: ({true_next_xy[0]:+.3f}, {true_next_xy[1]:+.3f}) m\n"
                f"Pred yaw: {math.degrees(pred['next_yaw']):+.2f}°\n"
                f"Real yaw: {math.degrees(after['block_yaw']):+.2f}°\n\n"
                f"Pred contact: {pred['contact']}\n"
                f"Real contact: {contact_actual}\n"
                f"Contact correct: {pred['contact'] == contact_actual}\n"
                f"Geometry score: {pred['score_m']:+.5f} m"
            )
            self.status.set("OK — one real-observation-conditioned prediction and one "
                            "MuJoCo step. This is NOT free-running rollout.")
        except Exception as exc:
            # No invented result, no automatic retry, freeze actions until Reset.
            self.failed = True
            self.status.set(f"ACTION FAILED — {type(exc).__name__}: {exc}. "
                            "Actions disabled until Reset. No retry was performed.")
        finally:
            self._set_busy(False)

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.root.destroy()


def run_demo() -> None:
    # Expensive libraries and simulator are not imported during --self-test.
    import torch
    from src.models.integrated_pusht_m4 import IntegratedPushTWorldModel
    from src.simulation.pusht_env import PushTEnv

    checkpoints = (
        REPO_ROOT / "data/processed/m4_visual_v03/best_model.pt",
        REPO_ROOT / "data/processed/m4_yaw_v04/best_model.pt",
    )
    if not all(path.is_file() for path in checkpoints):
        raise FileNotFoundError("Frozen M4 checkpoint missing; do not retrain")
    model = IntegratedPushTWorldModel.restore_frozen(
        dynamics_checkpoint=checkpoints[0], yaw_checkpoint=checkpoints[1],
    )
    model.eval()
    model.requires_grad_(False)
    env = PushTEnv()
    try:
        root = tk.Tk()
        try:
            _app = InteractiveDemo(root, env, model, torch)
            root.mainloop()
        finally:
            try:
                if root.winfo_exists():
                    root.destroy()
            except tk.TclError:
                pass
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 live PushT one-step comparison")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--self-test", action="store_true",
                        help="Synthetic no-simulator QA: ONLY M5-04 authorized action")
    choice.add_argument("--run-demo", action="store_true",
                        help="Open live MuJoCo + Tk demo (requires separate M5-05 approval)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_demo()


if __name__ == "__main__":
    main()
