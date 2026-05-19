#!/usr/bin/env python3
"""Force JAX model into fp32 and compare to PT fp32.

If JAX fp32 == PT fp32, the bias is bf16-precision-related.
If JAX fp32 == JAX bf16 (the usual config), then JAX runs precision-stable.
"""
from __future__ import annotations

import os, sys, json, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

os.environ["OPENPI_PYTORCH_PRECISION"] = "float32"

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30
    cos = float((af @ bf) / denom)
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    print(f"  [{name:34s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}")


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

    # Force JAX dtype to fp32 too
    import dataclasses
    cfg_fp32 = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, dtype="float32"))

    print("=" * 100)
    print("Loading JAX in fp32 + PT in fp32")
    print("=" * 100)
    jax_policy = _pc.create_trained_policy(cfg_fp32, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg_fp32, CKPT_PT)
    jax_model = jax_policy._model
    pt_model = pt_policy._model
    pt_model.eval()

    # Print the actual dtype of JAX weights
    print("Checking JAX dtypes ...")
    from flax import nnx
    n_cast = 0
    for path, leaf in nnx.iter_graph(jax_model):
        if isinstance(leaf, nnx.Variable):
            if hasattr(leaf, 'value') and hasattr(leaf.value, 'dtype'):
                if leaf.value.dtype == jnp.bfloat16:
                    leaf.value = leaf.value.astype(jnp.float32)
                    n_cast += 1
    print(f"  Cast {n_cast} JAX leaves from bf16 -> fp32")

    fp32_count = 0; bf16_count = 0; other_count = 0
    for path, leaf in nnx.iter_graph(jax_model):
        if isinstance(leaf, nnx.Variable):
            if hasattr(leaf, 'value') and hasattr(leaf.value, 'dtype'):
                d = leaf.value.dtype
                if d == jnp.float32: fp32_count += 1
                elif d == jnp.bfloat16: bf16_count += 1
                else: other_count += 1
    print(f"  JAX params after cast: fp32={fp32_count}, bf16={bf16_count}, other={other_count}")
    print(f"  PT q_proj dtype: {pt_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype}")

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    noise_jax = jnp.asarray(noise_np)
    noise_pt = torch.from_numpy(noise_np).to("cuda")
    rng = jax.random.PRNGKey(0)

    print()
    print("=" * 100)
    print("Run JAX (fp32) + PT (fp32) 10-step diffusion")
    print("=" * 100)
    jax_actions = jax_model.sample_actions(rng, jax_obs, noise=noise_jax, num_steps=10)
    jax_actions_np = np.asarray(jax_actions.astype(jnp.float32))
    with torch.no_grad():
        pt_actions = pt_model.sample_actions("cuda", pt_obs, noise=noise_pt, num_steps=10)
    pt_actions_np = pt_actions.detach().float().cpu().numpy()

    print()
    _stats("JAX fp32 vs PT fp32 (raw actions)", jax_actions_np, pt_actions_np)
    jax_final = jax_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": jax_actions_np[0]})
    pt_final = pt_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": pt_actions_np[0]})
    _stats("JAX fp32 vs PT fp32 (post-unnorm)", np.asarray(jax_final["actions"]), np.asarray(pt_final["actions"]))


if __name__ == "__main__":
    main()
