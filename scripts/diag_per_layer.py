#!/usr/bin/env python3
"""Compare JAX vs PyTorch hidden states INSIDE the PaliGemma transformer,
layer by layer, on the prefix-only forward pass.

We already know embed_prefix produces identical outputs (cos=1.0) for both
backends with the embed_tokens re-tie fix. This script answers:

  - Does the prefix forward through 18 paligemma layers diverge?
  - If so, at which layer does cos drop and norm ratio explode?

This isolates whether the residual JAX/PT parity bug lives in the
paligemma stack, the gemma_expert stack, or the cross-attention between them.
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
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    nj = float(np.linalg.norm(a))
    npt = float(np.linalg.norm(b))
    ratio = npt / (nj + 1e-12)
    print(
        f"  {name:25s}  cos={cos:+.6f}  |jax|={nj:>10.3f}  |pt|={npt:>10.3f}  ratio={ratio:.4f}  max|diff|={float(np.max(np.abs(a-b))):.4f}"
    )


def _load_obs():
    import subprocess
    import cv2
    import pyarrow.parquet as pq

    ds = Path("/root/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parquet = sorted(ds.glob("data/chunk-*/episode_*.parquet"))[0]
    table = pq.read_table(parquet)
    row = 100
    state = np.asarray(table.column("observation.state")[row].as_py(), dtype=np.float32)
    fidx = int(table.column("frame_index")[row].as_py())
    tidx = int(table.column("task_index")[row].as_py())
    tasks = {
        json.loads(l)["task_index"]: json.loads(l)["task"]
        for l in (ds / "meta" / "tasks.jsonl").read_text().splitlines()
    }
    prompt = tasks.get(tidx, "do the task")
    chunk, ep = parquet.parent.name, parquet.stem
    images = {}
    for cam_in, cam_out in (
        ("ego", "cam_high"),
        ("left_wrist", "cam_left_wrist"),
        ("right_wrist", "cam_right_wrist"),
    ):
        vp = ds / "videos" / chunk / f"observation.images.{cam_in}" / f"{ep}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            name = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vp),
             "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1", "-f", "image2", name],
            capture_output=True, timeout=15, check=True,
        )
        frame = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images[cam_out] = np.transpose(frame, (2, 0, 1)).astype(np.uint8)
    return {"state": state, "images": images, "prompt": prompt}


def run_jax_layer_by_layer(jax_model, jax_obs_p):
    """Run prefix through paligemma layers in JAX, returning per-layer outputs.

    JAX's llm uses nn.scan over the 18 Block layers, so it doesn't naturally
    expose per-layer hidden states. We hand-roll the equivalent computation by
    directly calling the layers Module bound to JAX's params.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0 as _pi0
    from openpi.models import gemma as _gemma

    prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(jax_obs_p)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    per_layer: list = []
    orig_call = _gemma.Block.__call__

    def patched_call(self, xs, kv_cache, positions_, attn_mask, adarms_cond_, deterministic=True):  # noqa: FBT002
        out = orig_call(self, xs, kv_cache, positions_, attn_mask, adarms_cond_, deterministic)
        try:
            arr = out[0][0]
            if arr is not None:
                per_layer.append(np.asarray(arr.astype(jnp.float32)))
        except Exception as e:
            print(f"  (patched_call capture failed: {e})")
        return out

    _gemma.Block.__call__ = patched_call
    try:
        with jax.disable_jit():
            out_list, _kv = jax_model.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
            )
    finally:
        _gemma.Block.__call__ = orig_call

    prefix_out = out_list[0]
    return {
        "embed_prefix": np.asarray(prefix_tokens.astype(jnp.float32)),
        "per_layer_outputs": per_layer,
        "final_hidden_state": np.asarray(prefix_out.astype(jnp.float32)),
        "positions": np.asarray(positions),
        "mask": np.asarray(prefix_mask),
    }


def run_pt_layer_by_layer(pt_model, pt_obs):
    """Run prefix through paligemma layers in PT, capturing per-layer outputs.

    Mirrors the structure used by compute_prefix_kv_cache, but additionally
    stores the hidden_states after EACH layer for layerwise comparison.
    """
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    from transformers.models.gemma import modeling_gemma as _hf_gemma

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _state = pt_model._preprocess_observation(pt_obs, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        paligemma_lm = pt_model.paligemma_with_expert.paligemma.language_model
        num_layers = paligemma_lm.config.num_hidden_layers

        if paligemma_lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = pt_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        hidden_states = prefix_embs
        layer_outputs = [hidden_states.detach().float().cpu().numpy().copy()]

        for layer_idx in range(num_layers):
            layer = paligemma_lm.layers[layer_idx]
            normed_hidden, _ = layer.input_layernorm(hidden_states, cond=None)
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)

            dummy = torch.zeros(
                query_states.shape[0], query_states.shape[2], query_states.shape[-1],
                device=query_states.device, dtype=query_states.dtype,
            )
            cos, sin = paligemma_lm.rotary_emb(dummy, position_ids)
            query_states, key_states = _hf_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            batch_size = query_states.shape[0]
            seq_len = query_states.shape[2]
            num_kv_groups = layer.self_attn.num_key_value_groups
            key_expanded = (
                key_states[:, :, None, :, :]
                .expand(batch_size, key_states.shape[1], num_kv_groups, seq_len, key_states.shape[-1])
                .reshape(batch_size, -1, seq_len, key_states.shape[-1])
            )
            value_expanded = (
                value_states[:, :, None, :, :]
                .expand(batch_size, value_states.shape[1], num_kv_groups, seq_len, value_states.shape[-1])
                .reshape(batch_size, -1, seq_len, value_states.shape[-1])
            )

            # Force JAX-equivalent attention: compute Q @ K^T in FP32, softmax in
            # FP32, downcast probs to bf16 before applying to V.
            import os as _os
            use_fp32_attn = _os.environ.get("OPENPI_PT_FP32_ATTN", "0") == "1"
            if use_fp32_attn:
                q_f32 = query_states.float()
                k_f32 = key_expanded.float()
                v_f32 = value_expanded.float()
                attn_logits = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * layer.self_attn.scaling
                attn_logits = attn_logits + prefix_att_2d_masks_4d.float()
                attn_probs = torch.softmax(attn_logits, dim=-1)
                att_output = torch.matmul(attn_probs, v_f32).to(query_states.dtype)
            else:
                att_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, key_expanded, value_expanded,
                    attn_mask=prefix_att_2d_masks_4d.to(query_states.dtype),
                    dropout_p=0.0, is_causal=False, scale=layer.self_attn.scaling,
                )
            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
            if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(att_output)
            out_emb = hidden_states + out_emb
            after_first_residual = out_emb.clone()

            out_emb, _ = layer.post_attention_layernorm(out_emb, cond=None)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)
            hidden_states = after_first_residual + out_emb

            layer_outputs.append(hidden_states.detach().float().cpu().numpy().copy())

        # Final norm
        hidden_states_final, _ = paligemma_lm.norm(hidden_states, cond=None)
        layer_outputs.append(hidden_states_final.detach().float().cpu().numpy().copy())

        return {
            "embed_prefix": prefix_embs.detach().float().cpu().numpy(),
            "per_layer_outputs": layer_outputs,  # 20 entries: input, after_layer_0..17, after_final_norm
            "final_hidden_state": hidden_states_final.detach().float().cpu().numpy(),
        }


def main() -> int:
    raw_obs = _load_obs()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _pc
    from openpi.models import model as _model
    import jax
    import jax.numpy as jnp
    import torch

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

    print("\n=== Running JAX prefix forward (all 18 paligemma layers + final_norm) ===")
    jax_out = run_jax_layer_by_layer(jax_policy._model, jax_obs_p)
    print(f"JAX final hidden_state shape: {jax_out['final_hidden_state'].shape}  |norm|={np.linalg.norm(jax_out['final_hidden_state']):.3f}")

    print("\n=== Running PT prefix forward (capturing every layer's output) ===")
    pt_out = run_pt_layer_by_layer(pt_policy._model, pt_obs)
    print(f"PT  final hidden_state shape: {pt_out['final_hidden_state'].shape}  |norm|={np.linalg.norm(pt_out['final_hidden_state']):.3f}")

    print("\n" + "=" * 100)
    print("COMPARISON: prefix hidden state at boundaries")
    print("=" * 100)
    _stats("embed_prefix (pre-LLM)", jax_out["embed_prefix"], pt_out["embed_prefix"])
    _stats("after 18 layers + norm", jax_out["final_hidden_state"], pt_out["final_hidden_state"])

    # Split language vs image tokens for the final state
    n_img = 768
    print("\n  Final hidden state, image vs language token slices:")
    _stats(" image slice (final)", jax_out["final_hidden_state"][:, :n_img], pt_out["final_hidden_state"][:, :n_img])
    _stats(" lang  slice (final)", jax_out["final_hidden_state"][:, n_img:], pt_out["final_hidden_state"][:, n_img:])

    # Per layer in PT: print norm progression so we can see where things blow up vs JAX final
    print("\n" + "=" * 100)
    print("Per-layer cosine sim (JAX vs PT) and norm progression")
    print("=" * 100)

    jax_per = jax_out["per_layer_outputs"]
    # PT per_layer_outputs has 20 entries: [embed, after_layer_0..17, post-final-norm]
    # JAX per_layer_outputs (from patched Block) has 18 entries: after_layer_0..17
    print(f"  (captured {len(jax_per)} JAX layer outputs, {len(pt_out['per_layer_outputs'])} PT layer outputs)")

    # PT per-layer norms (no JAX comparison since JAX capture is hard to make work)
    print("\n  PT per-layer norms (look for explosion):")
    for i, lh in enumerate(pt_out["per_layer_outputs"]):
        tag = "embed" if i == 0 else ("post-final-norm" if i == len(pt_out["per_layer_outputs"]) - 1 else f"after layer {i-1:>2}")
        print(f"    PT {tag:18s}  |norm|={np.linalg.norm(lh):>10.3f}  max|x|={float(np.max(np.abs(lh))):.4f}")

    # Align JAX layers with PT layers 1..18 (the "after layer N" entries)
    pt_after_layers = pt_out["per_layer_outputs"][1:1 + len(jax_per)]
    n_img = 768
    print(f"\n  {'layer':>6}  {'cos(all)':>10}  {'|jax|':>10}  {'|pt|':>10}  {'ratio':>7}  {'cos(img)':>10}  {'cos(lang)':>10}")
    for i, (j, p) in enumerate(zip(jax_per, pt_after_layers)):
        cos_all = float((j.flatten() @ p.flatten()) / (np.linalg.norm(j.flatten()) * np.linalg.norm(p.flatten()) + 1e-12))
        nj = float(np.linalg.norm(j)); np_ = float(np.linalg.norm(p))
        # Slice
        j_img, p_img = j[:, :n_img], p[:, :n_img]
        j_lang, p_lang = j[:, n_img:], p[:, n_img:]
        cos_img = float((j_img.flatten() @ p_img.flatten()) / (np.linalg.norm(j_img.flatten()) * np.linalg.norm(p_img.flatten()) + 1e-12))
        cos_lang = float((j_lang.flatten() @ p_lang.flatten()) / (np.linalg.norm(j_lang.flatten()) * np.linalg.norm(p_lang.flatten()) + 1e-12))
        print(f"  layer{i:>2}  {cos_all:>+10.6f}  {nj:>10.1f}  {np_:>10.1f}  {np_/(nj+1e-12):>7.3f}  {cos_img:>+10.6f}  {cos_lang:>+10.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
