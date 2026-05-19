#!/usr/bin/env python3
"""Per-step comparison of JAX vs PT inference WITH LoRA.

Walks the full diffusion loop side-by-side. At each step we compare:
  - v_t (output of action_out_proj)
  - x_t after Euler step

Also compares prefix_tokens and suffix_tokens once before the loop.

Goal: identify where the 8% magnitude bias accumulates.
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
        print(f"  [{name}] SHAPE mismatch: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30
    cos = float((af @ bf) / denom)
    nj = float(np.linalg.norm(a)); npt = float(np.linalg.norm(b))
    diff = a - b
    print(f"  [{name:32s}] cos={cos:+.6f}  |jax|={nj:>9.3f}  |pt|={npt:>9.3f}  ratio={npt/(nj+1e-12):.4f}  max|d|={float(np.max(np.abs(diff))):.5f}")


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
    import einops

    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")

    print("=" * 100)
    print("Loading JAX + PT policies")
    print("=" * 100)
    jax_policy = _pc.create_trained_policy(cfg, CKPT_JAX)
    pt_policy = _pc.create_trained_policy(cfg, CKPT_PT)

    jax_model = jax_policy._model
    pt_model = pt_policy._model
    pt_model.eval()

    inputs = jax_policy._input_transform(raw_obs)
    jax_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
    pt_inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x))[None].to("cuda"), inputs)
    jax_obs = _model.Observation.from_dict(jax_inputs)
    pt_obs = _model.Observation.from_dict(pt_inputs)

    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)

    print()
    print("=" * 100)
    print("Step A: embed_prefix")
    print("=" * 100)
    with jax.disable_jit():
        jax_prefix_tokens, jax_prefix_mask, jax_prefix_ar = jax_model.embed_prefix(jax_obs)
    jax_prefix_tokens_np = np.asarray(jax_prefix_tokens.astype(jnp.float32))

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = pt_model._preprocess_observation(pt_obs, train=False)
        pt_prefix_tokens, pt_prefix_mask, pt_prefix_ar = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    pt_prefix_tokens_np = pt_prefix_tokens.detach().float().cpu().numpy()
    _stats("prefix_tokens", jax_prefix_tokens_np, pt_prefix_tokens_np)

    jm = np.asarray(jax_prefix_mask)
    pm = pt_prefix_mask.detach().cpu().numpy()
    print(f"  mask shapes jax={jm.shape} pt={pm.shape}  equal={np.array_equal(jm, pm)}")

    print()
    print("=" * 100)
    print("Step B: embed_suffix (t=1.0)")
    print("=" * 100)
    t0 = jnp.array([1.0])
    noise_jax = jnp.asarray(noise_np)
    with jax.disable_jit():
        jax_suffix_tokens, jax_suffix_mask, jax_suffix_ar, jax_adarms = jax_model.embed_suffix(
            jax_obs, noise_jax, t0
        )
    jax_suffix_np = np.asarray(jax_suffix_tokens.astype(jnp.float32))
    jax_adarms_np = np.asarray(jax_adarms.astype(jnp.float32)) if jax_adarms is not None else None

    with torch.no_grad():
        noise_pt = torch.from_numpy(noise_np).to("cuda")
        t_pt = torch.tensor([1.0], device="cuda")
        pt_suffix_tokens, pt_suffix_mask, pt_suffix_ar, pt_adarms = pt_model.embed_suffix(state, noise_pt, t_pt)
    pt_suffix_np = pt_suffix_tokens.detach().float().cpu().numpy()
    pt_adarms_np = pt_adarms.detach().float().cpu().numpy() if pt_adarms is not None else None

    _stats("suffix_tokens", jax_suffix_np, pt_suffix_np)
    if jax_adarms_np is not None and pt_adarms_np is not None:
        _stats("adarms_cond", jax_adarms_np, pt_adarms_np)

    print()
    print("=" * 100)
    print("Step C: KV cache fill + first denoise step")
    print("=" * 100)
    from openpi.models.pi0 import make_attn_mask
    jax_prefix_attn_mask = make_attn_mask(jax_prefix_mask, jax_prefix_ar)
    jax_prefix_pos = jnp.cumsum(jax_prefix_mask, axis=1) - 1
    with jax.disable_jit():
        _, jax_kv_cache = jax_model.PaliGemma.llm(
            [jax_prefix_tokens, None], mask=jax_prefix_attn_mask, positions=jax_prefix_pos
        )

    with torch.no_grad():
        pt_kv_cache = pt_model.compute_prefix_kv_cache(pt_prefix_tokens, pt_prefix_mask, pt_prefix_ar)

    # CRITICAL TEST: inject JAX's prefix tokens into PT and rerun the cache fill.
    # If KV cache then matches JAX, the bug is in embedding. Otherwise it's in the layer stack.
    with torch.no_grad():
        jax_prefix_as_pt = torch.from_numpy(jax_prefix_tokens_np).to(pt_prefix_tokens.device).to(pt_prefix_tokens.dtype)
        pt_kv_cache_inject = pt_model.compute_prefix_kv_cache(jax_prefix_as_pt, pt_prefix_mask, pt_prefix_ar)

    # Compare prefix KV cache (JAX vs PT)
    print()
    print("  --- Prefix KV cache comparison ---")
    # JAX kv_cache structure: (K, V) tuple with leading dim = num_layers due to scan
    if isinstance(jax_kv_cache, tuple) and len(jax_kv_cache) == 2:
        jk_all, jv_all = jax_kv_cache  # (L, B, T, H, D)
        jk_np_all = np.asarray(jk_all.astype(jnp.float32))
        jv_np_all = np.asarray(jv_all.astype(jnp.float32))
        n_layers_jax = jk_np_all.shape[0]
        n_layers_pt = len(pt_kv_cache)
        print(f"  JAX layers: {n_layers_jax}, PT layers: {n_layers_pt}")
        for L in [0, 1, 5, 9, 13, 17]:
            if L >= min(n_layers_jax, n_layers_pt):
                continue
            pk = pt_kv_cache[L][0].detach().float().cpu().numpy()
            pv = pt_kv_cache[L][1].detach().float().cpu().numpy()
            jk = jk_np_all[L]; jv = jv_np_all[L]
            # PT shape: (B, H, T, D); JAX shape: (B, T, H, D) — transpose for comparison
            if jk.shape != pk.shape:
                # Try (B, T, H, D) → (B, H, T, D)
                if jk.shape[1] == pk.shape[2] and jk.shape[2] == pk.shape[1]:
                    jk = jk.transpose(0, 2, 1, 3)
                    jv = jv.transpose(0, 2, 1, 3)
            _stats(f"KV K layer {L}", jk, pk)
            _stats(f"KV V layer {L}", jv, pv)
            pk_i = pt_kv_cache_inject[L][0].detach().float().cpu().numpy()
            pv_i = pt_kv_cache_inject[L][1].detach().float().cpu().numpy()
            _stats(f"KV K layer {L} (inj)", jk, pk_i)
            _stats(f"KV V layer {L} (inj)", jv, pv_i)

    # JAX denoise step at t=1.0
    jax_suffix_attn_mask = make_attn_mask(jax_suffix_mask, jax_suffix_ar)
    jax_prefix_attn_s = einops.repeat(jax_prefix_mask, "b p -> b s p", s=jax_suffix_tokens.shape[1])
    jax_full_mask = jnp.concatenate([jax_prefix_attn_s, jax_suffix_attn_mask], axis=-1)
    jax_step_pos = jnp.sum(jax_prefix_mask, axis=-1)[:, None] + jnp.cumsum(jax_suffix_mask, axis=-1) - 1
    with jax.disable_jit():
        (jpo, jso), _ = jax_model.PaliGemma.llm(
            [None, jax_suffix_tokens], mask=jax_full_mask, positions=jax_step_pos,
            kv_cache=jax_kv_cache, adarms_cond=[None, jax_adarms],
        )
        jvt = jax_model.action_out_proj(jso[:, -horizon:])
    jax_suffix_out_np = np.asarray(jso.astype(jnp.float32))
    jax_vt0 = np.asarray(jvt.astype(jnp.float32))

    captured_pt_suffix_out = {}
    def _capture_pt(module, args, _kwargs=None):
        captured_pt_suffix_out["input"] = args[0].detach().float().cpu().numpy()
    hook = pt_model.action_out_proj.register_forward_pre_hook(_capture_pt, with_kwargs=False)

    # Per-layer hooks on PT gemma_expert
    pt_layer_outputs = {}
    pt_layer_hooks = []
    ge_layers = pt_model.paligemma_with_expert.gemma_expert.model.layers
    for i, layer in enumerate(ge_layers):
        def _make_hook(idx):
            def _h(_mod, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                pt_layer_outputs[idx] = out.detach().float().cpu().numpy()
            return _h
        pt_layer_hooks.append(layer.register_forward_hook(_make_hook(i)))

    try:
        with torch.no_grad():
            pt_v_t = pt_model.denoise_step_with_cache(state, pt_kv_cache, pt_prefix_mask, noise_pt, t_pt)
    finally:
        hook.remove()
        for h in pt_layer_hooks:
            h.remove()
    pt_vt0 = pt_v_t.detach().float().cpu().numpy()
    pt_suffix_out_np = captured_pt_suffix_out["input"]
    jax_suffix_pre_proj = jax_suffix_out_np[:, -horizon:]
    _stats("suffix_out (pre-out_proj)", jax_suffix_pre_proj, pt_suffix_out_np)
    _stats("v_t @ t=1.0", jax_vt0, pt_vt0)

    print()
    print("=" * 100)
    print("Step D: Full 10-step diffusion loop, per-step v_t comparison")
    print("=" * 100)
    num_steps = 10
    dt = -1.0 / num_steps
    x_t_jax = jnp.asarray(noise_np)
    x_t_pt = torch.from_numpy(noise_np).to("cuda")
    t_val = 1.0
    for step_idx in range(num_steps):
        with jax.disable_jit():
            jst, jsm, jsar, jacd = jax_model.embed_suffix(jax_obs, x_t_jax, jnp.array([t_val]))
            jsam = make_attn_mask(jsm, jsar)
            jpam = einops.repeat(jax_prefix_mask, "b p -> b s p", s=jst.shape[1])
            jfam = jnp.concatenate([jpam, jsam], axis=-1)
            jpos = jnp.sum(jax_prefix_mask, axis=-1)[:, None] + jnp.cumsum(jsm, axis=-1) - 1
            (_, jso2), _ = jax_model.PaliGemma.llm(
                [None, jst], mask=jfam, positions=jpos, kv_cache=jax_kv_cache, adarms_cond=[None, jacd]
            )
            jvt2 = jax_model.action_out_proj(jso2[:, -horizon:])
        with torch.no_grad():
            t_pt = torch.tensor([t_val], device="cuda")
            pvt2 = pt_model.denoise_step_with_cache(state, pt_kv_cache, pt_prefix_mask, x_t_pt, t_pt)

        jvt_np = np.asarray(jvt2.astype(jnp.float32))
        pvt_np = pvt2.detach().float().cpu().numpy()
        af, bf = jvt_np.flatten(), pvt_np.flatten()
        cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
        ratio = float(np.linalg.norm(pvt_np) / (np.linalg.norm(jvt_np) + 1e-12))
        # First-token components
        jvt0 = jvt_np[0, 0, :8]
        pvt0 = pvt_np[0, 0, :8]
        print(f"  step {step_idx:2d}  t={t_val:.3f}  cos={cos:+.5f}  |jvt|={np.linalg.norm(jvt_np):.3f}  |pvt|={np.linalg.norm(pvt_np):.3f}  ratio={ratio:.4f}  max|d|={float(np.max(np.abs(jvt_np-pvt_np))):.4f}")

        x_t_jax = x_t_jax + dt * jvt2
        x_t_pt = x_t_pt + dt * pvt2
        t_val += dt

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)
    x_t_jax_np = np.asarray(x_t_jax.astype(jnp.float32))
    x_t_pt_np = x_t_pt.detach().float().cpu().numpy()
    _stats("final x_0", x_t_jax_np, x_t_pt_np)
    _stats("first action [:16]", x_t_jax_np[:, 0, :16], x_t_pt_np[:, 0, :16])

    # Output transform comparison
    jax_final = jax_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": x_t_jax_np[0]})
    pt_final = pt_policy._output_transform({"state": np.asarray(inputs["state"]), "actions": x_t_pt_np[0]})
    _stats("post-unnorm actions", np.asarray(jax_final["actions"]), np.asarray(pt_final["actions"]))
    j3 = jax_final["actions"][0, 3]; p3 = pt_final["actions"][0, 3]
    print(f"  shoulder pitch (joint 3) action[0]: jax={j3:.5f} pt={p3:.5f} diff={p3-j3:+.5f}")


if __name__ == "__main__":
    main()
