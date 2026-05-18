#!/usr/bin/env python3
"""Compare JAX vs PT FULL diffusion loop (10 denoise steps) with IDENTICAL
inputs. If single-step cos=0.995 holds and the loop is correct, the final
action chunk should also have cos>=0.9 with JAX.

Also dumps per-step v_t norms so we can see if error compounds.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    print(f"  [{name:18s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}  max|d|={float(np.max(np.abs(a-b))):.4f}")


def _load_obs():
    import subprocess, cv2
    import pyarrow.parquet as pq
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            name = tmp.name
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp), "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1", "-f", "image2", name], capture_output=True, timeout=15, check=True)
        frame = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def main():
    raw_obs = _load_obs()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax, jax.numpy as jnp, torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    print("Loading both policies ...")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)

    inputs = jax_policy._input_transform(raw_obs)
    print(f"After input transform: state[:8]={inputs['state'][:8]}")
    print(f"  tokenized_prompt[:10]={inputs['tokenized_prompt'][:10]}")

    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    # SAME fixed noise for both
    horizon = jax_policy._model.action_horizon
    adim = jax_policy._model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    print(f"\nFixed noise: |{np.linalg.norm(noise_np):.3f}|  range=[{noise_np.min():.3f}, {noise_np.max():.3f}]")

    # JAX: full sample_actions
    print("\n=== JAX full sample_actions ===")
    noise_jax = jnp.asarray(noise_np)
    rng = jax.random.PRNGKey(0)
    jax_actions = jax_policy._model.sample_actions(rng, jax.tree.map(jnp.asarray,
        jax.tree.map(lambda x: x, jax_obs)), noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    print(f"JAX actions: shape={jax_actions_np.shape}  |norm|={np.linalg.norm(jax_actions_np):.3f}")

    # PT: full sample_actions
    print("\n=== PT full sample_actions ===")
    noise_pt = torch.from_numpy(noise_np).to("cuda")
    pt_model = pt_policy._model
    pt_model.eval()
    with torch.no_grad():
        pt_actions = pt_model.sample_actions("cuda", pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_np = pt_actions.detach().float().cpu().numpy()
    print(f"PT  actions: shape={pt_actions_np.shape}  |norm|={np.linalg.norm(pt_actions_np):.3f}")

    print("\n" + "=" * 100)
    print("COMPARISON: final action chunk after 10 denoise steps with fixed noise")
    print("=" * 100)
    _stats("actions", jax_actions_np, pt_actions_np)

    # Slice: first 16 action dims (the real robot DOFs)
    _stats("actions[:, :, :16]", jax_actions_np[:, :, :16], pt_actions_np[:, :, :16])
    _stats("first action [0,0]", jax_actions_np[:, 0, :16], pt_actions_np[:, 0, :16])

    print("\n  JAX first action [:16]:  ", jax_actions_np[0, 0, :16])
    print("  PT  first action [:16]:  ", pt_actions_np[0, 0, :16])
    print("  diff           [:16]:  ", pt_actions_np[0, 0, :16] - jax_actions_np[0, 0, :16])

    # Now also check what the POLICY would return after the OUTPUT TRANSFORMS (unnorm + repack)
    print("\n" + "=" * 100)
    print("COMPARISON: after policy server's output transforms (unnormalize + repack)")
    print("=" * 100)
    jax_output = {"state": np.asarray(inputs["state"]), "actions": jax_actions_np[0]}
    pt_output = {"state": np.asarray(inputs["state"]), "actions": pt_actions_np[0]}
    jax_final = jax_policy._output_transform(jax_output)
    pt_final = pt_policy._output_transform(pt_output)
    jax_a = jax_final.get("actions", None)
    pt_a = pt_final.get("actions", None)
    if jax_a is not None and pt_a is not None:
        ja, pa = np.asarray(jax_a), np.asarray(pt_a)
        _stats("post-output-transform actions", ja, pa)
        print(f"\n  JAX post-transform first action: {ja[0, :16]}")
        print(f"  PT  post-transform first action: {pa[0, :16]}")


if __name__ == "__main__":
    main()
