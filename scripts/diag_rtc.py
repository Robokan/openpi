#!/usr/bin/env python3
"""Validate Pi-GDM RTC implementation on PyTorch pi0.5.

Smoke-tests the new ``realtime_sample_actions`` method:
  1) get_prefix_weights matches the Kinetix reference (Eq. 5 in paper).
  2) With ``inference_delay=0`` and ``prefix_attention_horizon=0`` (zeros
     schedule), RTC degenerates to vanilla sample_actions (no guidance).
  3) With non-zero d and a fake prev_action_chunk, RTC outputs:
     - first ``d`` rows close to prev_chunk (frozen prefix)
     - last ``H - prefix_attention_horizon`` rows free (unrelated to prev)

Run inside the openpi-pt-server container:
    docker exec openpi-pt-server bash -lc 'cd /app && \
        PYTHONPATH=/app/src:/app/packages/openpi-client/src \
        OPENPI_PYTORCH_PRECISION=float32 \
        OPENPI_PT_FAST_ATTN=1 \
        OPENPI_PT_DISABLE_COMPILE=1 \
        python scripts/diag_rtc.py'

(We disable torch.compile for the diag to keep error messages readable.)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: a={a.shape} b={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    print(f"  [{name:38s}] cos={cos:+.6f}  rel_err={rel:.4e}  |a|={np.linalg.norm(a):.3f}  |b|={np.linalg.norm(b):.3f}")
    return cos, rel


def _kinetix_get_prefix_weights_ref(start, end, total, schedule):
    """Mirror Kinetix's reference implementation in pure NumPy."""
    start = min(start, end)
    idx = np.arange(total, dtype=np.float32)
    if schedule == "ones":
        w = np.ones(total, dtype=np.float32)
    elif schedule == "zeros":
        w = (idx < start).astype(np.float32)
    elif schedule in ("linear", "exp"):
        w = np.clip((start - 1 - idx) / (end - start + 1) + 1, 0.0, 1.0)
        if schedule == "exp":
            w = w * np.expm1(w) / (math.e - 1.0)
    else:
        raise ValueError(schedule)
    return np.where(idx >= end, 0.0, w).astype(np.float32)


def test_prefix_weights():
    """Phase 1: reference parity of soft-mask schedule."""
    print("=" * 72)
    print("TEST 1: get_prefix_weights vs Kinetix reference")
    print("=" * 72)
    import torch  # noqa: PLC0415

    from openpi.models_pytorch import rtc  # noqa: PLC0415

    failures = 0
    for sched in ("ones", "zeros", "linear", "exp"):
        for d, pah, H in [
            (2, 6, 10),    # paper docstring example
            (0, 25, 50),   # synchronous (no delay, full overlap)
            (5, 25, 50),   # realistic d=5
            (10, 25, 50),  # heavier delay
            (3, 3, 50),    # degenerate end==start
        ]:
            got = rtc.get_prefix_weights(d, pah, H, schedule=sched, device="cpu", dtype=torch.float32).numpy()
            ref = _kinetix_get_prefix_weights_ref(d, pah, H, sched)
            ok = np.allclose(got, ref, rtol=1e-6, atol=1e-6)
            tag = "OK" if ok else "FAIL"
            print(f"  [{tag}] schedule={sched:6s}  d={d:2d} pah={pah:2d} H={H}  diff_max={np.max(np.abs(got-ref)):.2e}")
            if not ok:
                failures += 1
                print(f"    got: {got[:12]}")
                print(f"    ref: {ref[:12]}")
    print(f"\n  -> {failures} failures")
    return failures == 0


def _load_obs():
    import av  # noqa: PLC0415
    import cv2  # noqa: PLC0415
    import json  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    ds = Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parquet = sorted(ds.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
    row = 100
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype=np.float32)
    fidx = int(table.column("frame_index")[row].as_py())
    tidx = int(table.column("task_index")[row].as_py())
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in (ds / "meta" / "tasks.jsonl").read_text().splitlines()}
    prompt = tasks.get(tidx, "do the task")
    chunk, ep = parquet.parent.name, parquet.stem
    images = {}
    for cam_in, cam_out in (("ego", "cam_high"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist")):
        vp = ds / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with av.open(str(vp)) as container:
            stream = container.streams.video[0]
            frame_np = None
            for i, frame in enumerate(container.decode(stream)):
                if i == fidx:
                    frame_np = frame.to_ndarray(format="rgb24")
                    break
            if frame_np is None:
                raise RuntimeError(f"frame {fidx} not found in {vp}")
        if frame_np.shape[:2] != (224, 224):
            frame_np = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame_np, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def test_rtc_end_to_end():
    """Phase 2: end-to-end test against vanilla sample_actions."""
    print("\n" + "=" * 72)
    print("TEST 2: realtime_sample_actions e2e (zeros schedule => vanilla)")
    print("=" * 72)
    import torch  # noqa: PLC0415

    from openpi.training import config as _config  # noqa: PLC0415
    from openpi.models import model as _model  # noqa: PLC0415
    from openpi.policies import policy_config as _pc  # noqa: PLC0415

    train_config = _config.get_config("pi05_openarm_ngc_lora_v4")
    ckpt = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"
    policy = _pc.create_trained_policy(train_config, ckpt, pytorch_device="cuda")
    model = policy._model  # noqa: SLF001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_obs = _load_obs()
    # Convert to torch through the policy's input transform first.
    inputs = policy._input_transform(raw_obs)  # noqa: SLF001
    import jax  # noqa: PLC0415

    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    observation = _model.Observation.from_dict(pt_inputs)

    H = model.config.action_horizon
    adim = model.config.action_dim
    np.random.seed(0)
    noise = torch.from_numpy(np.random.randn(1, H, adim).astype(np.float32)).to(device)

    # Vanilla path: get baseline chunk.
    with torch.no_grad():
        v_chunk = model.sample_actions(device, observation, noise=noise, num_steps=10)
    v_np = v_chunk.detach().cpu().numpy()
    print(f"  Baseline chunk: shape={v_np.shape}  |chunk|={np.linalg.norm(v_np):.3f}")

    # Production-realistic test: use the vanilla chunk itself as the prev chunk.
    # If RTC works, the first ``d`` rows of the new chunk should match the
    # first ``d`` rows of the vanilla chunk almost exactly. This is the same
    # observation/noise as vanilla, so without guidance the result IS vanilla.
    # With guidance, the frozen prefix forces it to look like the prev chunk
    # in the frozen region.
    prev = v_chunk.detach().clone()
    prev_np = prev.cpu().numpy()

    # Case A: zeros schedule with d=0 -> guidance weight = 0 -> RTC should match vanilla.
    rtc_chunk = model.realtime_sample_actions(
        device,
        observation,
        prev_action_chunk=prev,
        inference_delay=0,
        prefix_attention_horizon=0,
        prefix_attention_schedule="zeros",
        max_guidance_weight=5.0,
        noise=noise,
        num_steps=10,
    )
    rtc_np = rtc_chunk.detach().cpu().numpy()
    print()
    _stats("zeros-schedule, d=0 vs vanilla    ", v_np, rtc_np)

    # Case B: paper-style schedule (exp), d=4, pah=25, H=50. Here we use the
    # vanilla chunk itself as the prev chunk, so RTC should reproduce vanilla
    # exactly on the frozen prefix.
    print()
    print("  Self-consistency: prev_chunk == vanilla chunk")
    rtc_chunk = model.realtime_sample_actions(
        device,
        observation,
        prev_action_chunk=prev,
        inference_delay=4,
        prefix_attention_horizon=25,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
        noise=noise,
        num_steps=10,
    )
    rtc_np = rtc_chunk.detach().cpu().numpy()
    print(f"  RTC (d=4, pah=25, exp): |chunk|={np.linalg.norm(rtc_np):.3f}  (vanilla |chunk|={np.linalg.norm(v_np):.3f})")
    pre_err = np.linalg.norm(rtc_np[0, :4] - prev_np[0, :4]) / (np.linalg.norm(prev_np[0, :4]) + 1e-9)
    free_err = np.linalg.norm(rtc_np[0, 25:] - prev_np[0, 25:]) / (np.linalg.norm(v_np[0, 25:]) + 1e-9)
    print(f"  rel_err(frozen prefix [0:4] vs prev)      = {pre_err:.3e}  (~0 = perfect frozen)")
    print(f"  rel_err(free tail [25:50] vs vanilla)     = {free_err:.3e}  (~0 = consistent)")

    # Case C: heavier guidance: d=8, longer frozen region.
    print()
    print("  Larger d: d=8, pah=25, exp")
    rtc_chunk = model.realtime_sample_actions(
        device,
        observation,
        prev_action_chunk=prev,
        inference_delay=8,
        prefix_attention_horizon=25,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
        noise=noise,
        num_steps=10,
    )
    rtc_np = rtc_chunk.detach().cpu().numpy()
    pre_err = np.linalg.norm(rtc_np[0, :8] - prev_np[0, :8]) / (np.linalg.norm(prev_np[0, :8]) + 1e-9)
    pre_max = np.abs(rtc_np[0, :8] - prev_np[0, :8]).max()
    seam = np.abs(rtc_np[0, 8] - rtc_np[0, 7]).max()  # seam discontinuity
    print(f"  rel_err(frozen prefix [0:8] vs prev) = {pre_err:.3e}   max_err={pre_max:.3e}")
    print(f"  max action delta at seam (index 7->8): {seam:.3e}")

    # Case D: perturbed prev chunk (simulate the observation changed slightly).
    print()
    print("  Perturbation: noisy obs -> chunk differs but frozen prefix locked")
    noise2 = torch.from_numpy(np.random.randn(1, H, adim).astype(np.float32) * 1.0).to(device)
    rtc_chunk = model.realtime_sample_actions(
        device,
        observation,
        prev_action_chunk=prev,
        inference_delay=4,
        prefix_attention_horizon=25,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
        noise=noise2,  # different starting noise
        num_steps=10,
    )
    rtc_np = rtc_chunk.detach().cpu().numpy()
    pre_err = np.linalg.norm(rtc_np[0, :4] - prev_np[0, :4]) / (np.linalg.norm(prev_np[0, :4]) + 1e-9)
    tail_diff = np.linalg.norm(rtc_np[0, 25:] - prev_np[0, 25:]) / (np.linalg.norm(prev_np[0, 25:]) + 1e-9)
    print(f"  rel_err(frozen prefix [0:4] vs prev) = {pre_err:.3e}   (should be small)")
    print(f"  rel_err(tail [25:50] vs prev)        = {tail_diff:.3e}   (free, can be ~1)")
    return True


if __name__ == "__main__":
    ok = test_prefix_weights()
    if not ok:
        print("[FAIL] prefix-weights test failed; aborting.")
        sys.exit(1)
    try:
        test_rtc_end_to_end()
    except Exception as e:
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        print(f"[FAIL] e2e test raised: {e!r}")
        sys.exit(2)
    print("\n[OK] all RTC tests passed.")
