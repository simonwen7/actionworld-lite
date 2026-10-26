"""Request-local MuJoCo state, cached immutable PyTorch weights, no datasets."""
from __future__ import annotations

import base64
import os
import platform
import struct
import threading
import zlib
from pathlib import Path

import numpy as np

from . import model_adapter as adapter
from . import session

# Must precede the first mujoco import on Linux. OSMesa shared libs on Vercel
# are NOT verified; if missing, fail closed instead of a fake rendered image.
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")

_lock = threading.Lock()
_backend = None


def get_backend():
    """Lazily restore frozen weights once per worker. No per-user MjData reuse."""
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        adapter.assert_frozen_artifacts()
        import torch
        from src.models.integrated_pusht_m4 import IntegratedPushTWorldModel
        from src.simulation.pusht_env import PushTEnv
        import mujoco
        torch.set_num_threads(1)
        fitted = adapter.load_fitted()
        model = IntegratedPushTWorldModel.restore_frozen(
            dynamics_checkpoint=adapter.ROOT / adapter.DYNAMICS,
            yaw_checkpoint=adapter.ROOT / adapter.YAW,
        )
        model.eval().requires_grad_(False)
        _backend = (torch, mujoco, PushTEnv, model, fitted)
        return _backend


def _png(rgb: np.ndarray) -> str:
    """Lossless real MuJoCo RGB to PNG without Pillow / external codec."""
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8 or not 1 <= a.shape[0] <= 512 or not 1 <= a.shape[1] <= 512:
        raise ValueError("Invalid MuJoCo RGB render")
    h,w,_ = a.shape
    scanlines = b"".join(b"\x00"+np.ascontiguousarray(a[y]).tobytes() for y in range(h))
    def chunk(tag,payload):
        return struct.pack(">I",len(payload))+tag+payload+struct.pack(">I",zlib.crc32(tag+payload)&0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">2I5B",w,h,8,2,0,0,0))
    png += chunk(b"IDAT",zlib.compress(scanlines,6))+chunk(b"IEND",b"")
    return "data:image/png;base64,"+base64.b64encode(png).decode("ascii")


def _state(mujoco, env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    n = mujoco.mj_stateSize(env.model, spec)
    out = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, out, spec)
    return out


def _restore(mujoco, env, arr):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    n = mujoco.mj_stateSize(env.model, spec)
    if arr.shape != (n,) or not np.isfinite(arr).all():
        raise session.SessionError("Invalid restored physics state")
    mujoco.mj_setState(env.model, env.data, np.ascontiguousarray(arr), spec)
    mujoco.mj_forward(env.model, env.data)


def nstate():
    _,mj,Env,_,_=get_backend()
    env=Env()
    try:
        return mj.mj_stateSize(env.model,mj.mjtState.mjSTATE_INTEGRATION)
    finally:
        env.close()


def _actual(obs):
    return {"actual_xy_m":list(obs["block_xy"]),"actual_yaw_rad":float(obs["block_yaw"]),
            "pusher_xy_m":list(obs["pusher_xy"]),"actual_contact":bool(obs["pusher_contact"])}


def reset_scene():
    """Start two TRUE observed RGB frames, including an unscored zero step."""
    _, mj, Env, _, _=get_backend()
    env=Env()
    try:
        env.reset()
        prev=env.render()
        env.step((0.0,0.0))
        # mj_step integrates qpos after the final forward pass; synchronize
        # derived geometry and contact before observing or storing RGB.
        mj.mj_forward(env.model,env.data)
        after=env.observe()
        current=env.render()
        state=_state(mj,env)
        response={"scored_actions":0,"physics_time_s":float(after["time"]),
                  "real_rgb_png":_png(current),"actual":_actual(after),
                  "prediction":None,"metrics":None,
                  "notes":"Two actual pre-action RGB observations; startup zero step is unscored"}
        return response, state, adapter.stride64(prev), adapter.stride64(current)
    finally:
        env.close()


def action_scene(saved_state,previous,current,direction,mm):
    torch,mj,Env,model,fitted=get_backend()
    offsets={"up":(0.0,1.0),"down":(0.0,-1.0),"left":(-1.0,0.0),"right":(1.0,0.0)}
    if type(mm) is not int or mm not in (5,10,20) or direction not in offsets:
        raise ValueError("Only valid direction and 5,10,20 mm steps are accepted")
    delta=tuple(axis * mm / 1000.0 for axis in offsets[direction])
    env=Env()
    try:
        _restore(mj,env,saved_state)
        before=env.observe()
        prediction=adapter.infer(model,fitted,previous,current,delta,torch)
        env.step(delta)
        # Match the saved integration state to the actual rendered/observed
        # post-action scene, including geometry-dependent contact metrics.
        mj.mj_forward(env.model,env.data)
        after=env.observe()
        new_rgb=env.render()
        metrics=adapter.score_step(prediction,before,after)
        response={"physics_time_s":float(after["time"]),"real_rgb_png":_png(new_rgb),
                  "actual":_actual(after),"prediction":prediction,"metrics":metrics,
                  "ic01_forward_calls":1,"physics_step_calls":1,
                  "prediction_source":"two real RGB before action",
                  "step_size_mm":mm,"direction":direction,"action_m":list(delta)}
        return response,_state(mj,env),current.copy(),adapter.stride64(new_rgb)
    finally:
        env.close()


def inspect_restoration_once():
    """Operator-only Step 2/3 real CPU parity check, NO TRAIN/VAL/TEST.

    Reports per-action differences using the actual simulator from new resets.
    RGB cross-request equivalence remains a critical acceptance gate.
    """
    _,mj,Env,_,_=get_backend()
    a,b=Env(),Env()
    try:
        a.reset();a.render();a.step((0.,0.))
        mj.mj_forward(a.model,a.data)
        a.render()
        state=_state(mj,a)
        _restore(mj,b,state)
        base_a=a.observe(); base_b=b.observe()
        rgb_a=a.render();rgb_b=b.render()
        a.step((.01,0));b.step((.01,0))
        mj.mj_forward(a.model,a.data)
        mj.mj_forward(b.model,b.data)
        result_a=a.observe();result_b=b.observe()
        end_a=a.render();end_b=b.render()
        return {
            "baseline_xy_max_abs_m":float(np.max(np.abs(np.array(base_a["block_xy"])-np.array(base_b["block_xy"])))),
            "base_rgb_equal":bool(np.array_equal(rgb_a,rgb_b)),
            "next_xy_max_abs_m":float(np.max(np.abs(np.array(result_a["block_xy"])-np.array(result_b["block_xy"])))),
            "next_rgb_equal":bool(np.array_equal(end_a,end_b)),
            "next_contact_equal":result_a["pusher_contact"]==result_b["pusher_contact"],
        }
    finally:
        a.close();b.close()
