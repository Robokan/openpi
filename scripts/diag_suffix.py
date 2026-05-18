#!/usr/bin/env python3
"""Compare JAX vs PT *suffix* output for ONE denoise step with identical
inputs: same observation, same noise, same timestep.

This pinpoints whether the residual parity bug is in the gemma_expert
forward / suffix path / action_out_proj, by removing variance from the
diffusion sampling loop and from the input data.
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

FIXED_T = 0.5  # mid-diffusion


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    print(f"  [{name:35s}] cos={cos:+.6f}  |jax|={float(np.linalg.norm(a)):>10.3f}  |pt|={float(np.linalg.norm(b)):>10.3f}  ratio={float(np.linalg.norm(b)/(np.linalg.norm(a)+1e-12)):.4f}  max|diff|={float(np.max(np.abs(a-b))):.4f}")


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
    from openpi.models import pi0 as _pi0
    import jax, jax.numpy as jnp, einops, torch

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    print("Loading both policies ...")
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)
    jax_obs_p = _model.preprocess_observation(None, jax_obs, train=False)

    # Same noise and same timestep for both backends
    horizon = jax_policy._model.action_horizon
    adim = jax_policy._model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    print(f"\nFixed noise shape={noise_np.shape} norm={np.linalg.norm(noise_np):.3f}")
    print(f"Fixed t={FIXED_T}")

    # ===========================================================
    # JAX: one denoise step
    # ===========================================================
    print("\n=== JAX one denoise step ===")
    noise_jax = jnp.asarray(noise_np)
    t_jax = jnp.asarray(FIXED_T, dtype=jnp.float32)

    jax_model = jax_policy._model

    @jax.jit
    def jax_one_step(obs_p, noise, t):
        prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(obs_p)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_p = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = jax_model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions_p)
        # suffix forward
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = jax_model.embed_suffix(
            obs_p, noise, jnp.broadcast_to(t, (1,))
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask2 = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask2, suffix_attn_mask], axis=-1)
        positions_s = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_prefix_out, suffix_out), _ = jax_model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_s,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = jax_model.action_out_proj(suffix_out[:, -horizon:])
        return suffix_tokens, suffix_out, v_t, adarms_cond

    jax_suffix_in, jax_suffix_out, jax_v_t, jax_adarms = jax_one_step(jax_obs_p, noise_jax, t_jax)
    jax_suffix_in_np = np.asarray(jax_suffix_in.astype(jnp.float32))
    jax_suffix_out_np = np.asarray(jax_suffix_out.astype(jnp.float32))
    jax_v_t_np = np.asarray(jax_v_t.astype(jnp.float32))
    jax_adarms_np = np.asarray(jax_adarms.astype(jnp.float32))
    print(f"JAX suffix tokens in: shape={jax_suffix_in_np.shape}  |norm|={np.linalg.norm(jax_suffix_in_np):.3f}")
    print(f"JAX adarms_cond     : shape={jax_adarms_np.shape}  |norm|={np.linalg.norm(jax_adarms_np):.3f}  range=[{jax_adarms_np.min():.3f}, {jax_adarms_np.max():.3f}]")
    print(f"JAX suffix out      : shape={jax_suffix_out_np.shape}  |norm|={np.linalg.norm(jax_suffix_out_np):.3f}")
    print(f"JAX v_t             : shape={jax_v_t_np.shape}  |norm|={np.linalg.norm(jax_v_t_np):.3f}  range=[{jax_v_t_np.min():.3f}, {jax_v_t_np.max():.3f}]")

    # ===========================================================
    # PT: one denoise step using model paths
    # ===========================================================
    print("\n=== PT one denoise step ===")
    pt_model = pt_policy._model
    pt_model.eval()
    device = "cuda"
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = pt_model._preprocess_observation(pt_obs, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_kv_cache = pt_model.compute_prefix_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)

        x_t = torch.from_numpy(noise_np).to(device)
        t_pt = torch.tensor([FIXED_T], dtype=torch.float32, device=device)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pt_model.embed_suffix(state, x_t, t_pt)
        pt_suffix_in = suffix_embs.detach().float().cpu().numpy()
        pt_adarms = adarms_cond.detach().float().cpu().numpy() if adarms_cond is not None else None

        # Run the model's denoise_step_with_cache, but ALSO capture suffix_out before action_out_proj.
        # Easiest: call the same logic but split into pieces.
        # Reuse existing API: denoise_step_with_cache returns v_t directly. To get suffix_out we
        # need to monkey-patch action_out_proj.
        captured = {}
        orig_forward = pt_model.action_out_proj.forward
        def patched_forward(x):
            captured['suffix_out_for_proj'] = x.detach().float().cpu().numpy().copy()
            return orig_forward(x)
        pt_model.action_out_proj.forward = patched_forward
        try:
            v_t = pt_model.denoise_step_with_cache(
                state, prefix_kv_cache, prefix_pad_masks, x_t, t_pt,
            )
        finally:
            pt_model.action_out_proj.forward = orig_forward
        pt_v_t = v_t.detach().float().cpu().numpy()
        pt_suffix_out_for_proj = captured['suffix_out_for_proj']

    print(f"PT  suffix tokens in: shape={pt_suffix_in.shape}  |norm|={np.linalg.norm(pt_suffix_in):.3f}")
    if pt_adarms is not None:
        print(f"PT  adarms_cond     : shape={pt_adarms.shape}  |norm|={np.linalg.norm(pt_adarms):.3f}  range=[{pt_adarms.min():.3f}, {pt_adarms.max():.3f}]")
    print(f"PT  suffix-for-proj : shape={pt_suffix_out_for_proj.shape}  |norm|={np.linalg.norm(pt_suffix_out_for_proj):.3f}")
    print(f"PT  v_t             : shape={pt_v_t.shape}  |norm|={np.linalg.norm(pt_v_t):.3f}  range=[{pt_v_t.min():.3f}, {pt_v_t.max():.3f}]")

    print("\n" + "=" * 100)
    print("COMPARISON: one denoise step at t=0.5 with seed=42 noise, real observation")
    print("=" * 100)
    _stats("suffix_tokens (action_in_proj)", jax_suffix_in_np, pt_suffix_in)
    if pt_adarms is not None and jax_adarms_np is not None:
        _stats("adarms_cond                  ", jax_adarms_np, pt_adarms)
    _stats("suffix_out (last 50 tokens)  ", jax_suffix_out_np[:, -horizon:], pt_suffix_out_for_proj)
    _stats("v_t (action_out_proj output) ", jax_v_t_np, pt_v_t)

    # Per joint
    print(f"\n  v_t[0, action_step=0]:")
    print(f"    JAX: {jax_v_t_np[0, 0, :]}")
    print(f"    PT : {pt_v_t[0, 0, :]}")


if __name__ == "__main__":
    main()
