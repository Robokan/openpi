#!/usr/bin/env python3
"""Verify that the MODEL's compute_prefix_kv_cache + denoise path now
uses fp32 attention by running it on a real observation and comparing
the final FULL hidden_state of the prefix against JAX.

This differs from diag_per_layer.py in that it calls the *model's*
compute_prefix_kv_cache (the one the server uses), so we know the
fix is wired up end-to-end.
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
        print(f"  [{name}] SHAPE MISMATCH: jax={a.shape} pt={b.shape}")
        return
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    print(f"  [{name:35s}] cos={cos:+.6f}  |jax|={float(np.linalg.norm(a)):>10.3f}  |pt|={float(np.linalg.norm(b)):>10.3f}  max|diff|={float(np.max(np.abs(a-b))):.4f}")


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
    import jax, jax.numpy as jnp, torch

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

    # JAX: end-to-end prefix forward
    prefix_tokens, prefix_mask, prefix_ar_mask = jax_policy._model.embed_prefix(jax_obs_p)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    out_list, _kv = jax_policy._model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    jax_prefix_final = np.asarray(out_list[0].astype(jnp.float32))
    print(f"\nJAX prefix final (post final_norm): shape={jax_prefix_final.shape} |norm|={np.linalg.norm(jax_prefix_final):.3f}")

    # PT: call model's compute_prefix_kv_cache then run through final norm
    pt_model = pt_policy._model
    pt_model.eval()
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _state = pt_model._preprocess_observation(pt_obs, train=False)
        pt_prefix_embs, pt_prefix_pad_masks, pt_prefix_att_masks = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Recreate prefix layer-by-layer forward EXACTLY as compute_prefix_kv_cache does
        # but ALSO track the hidden_states (which compute_prefix_kv_cache doesn't return).
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        from openpi.models_pytorch.gemma_pytorch import fp32_attention
        from transformers.models.gemma import modeling_gemma as _hf_gemma
        paligemma_lm = pt_model.paligemma_with_expert.paligemma.language_model
        if paligemma_lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            pt_prefix_embs = pt_prefix_embs.to(dtype=torch.bfloat16)
        prefix_att_2d_masks = make_att_2d_masks(pt_prefix_pad_masks, pt_prefix_att_masks)
        prefix_att_2d_masks_4d = pt_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        position_ids = torch.cumsum(pt_prefix_pad_masks, dim=1) - 1
        hidden_states = pt_prefix_embs
        for layer_idx in range(paligemma_lm.config.num_hidden_layers):
            layer = paligemma_lm.layers[layer_idx]
            normed_hidden, _ = layer.input_layernorm(hidden_states, cond=None)
            input_shape = normed_hidden.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states = layer.self_attn.q_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(normed_hidden).view(hidden_shape).transpose(1, 2)
            dummy = torch.zeros(query_states.shape[0], query_states.shape[2], query_states.shape[-1], device=query_states.device, dtype=query_states.dtype)
            cos_, sin_ = paligemma_lm.rotary_emb(dummy, position_ids)
            query_states, key_states = _hf_gemma.apply_rotary_pos_emb(query_states, key_states, cos_, sin_, unsqueeze_dim=1)

            batch_size_ = query_states.shape[0]
            seq_len = query_states.shape[2]
            num_kv_groups = layer.self_attn.num_key_value_groups
            k_expanded = key_states[:, :, None, :, :].expand(batch_size_, key_states.shape[1], num_kv_groups, seq_len, key_states.shape[-1]).reshape(batch_size_, -1, seq_len, key_states.shape[-1])
            v_expanded = value_states[:, :, None, :, :].expand(batch_size_, value_states.shape[1], num_kv_groups, seq_len, value_states.shape[-1]).reshape(batch_size_, -1, seq_len, value_states.shape[-1])
            att_output = fp32_attention(query_states, k_expanded, v_expanded, additive_mask_4d=prefix_att_2d_masks_4d, scaling=layer.self_attn.scaling)
            head_dim = layer.self_attn.head_dim
            num_heads = layer.self_attn.config.num_attention_heads
            att_output = att_output.transpose(1, 2).reshape(batch_size_, seq_len, num_heads * head_dim)
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

        hidden_final, _ = paligemma_lm.norm(hidden_states, cond=None)
        pt_prefix_final = hidden_final.float().cpu().numpy()
        print(f"PT  prefix final (post final_norm): shape={pt_prefix_final.shape} |norm|={np.linalg.norm(pt_prefix_final):.3f}")

    print("\n" + "=" * 100)
    print("COMPARISON of final prefix hidden states (after fp32 attention fix)")
    print("=" * 100)
    _stats("full prefix final", jax_prefix_final, pt_prefix_final)
    _stats("image slice", jax_prefix_final[:, :768], pt_prefix_final[:, :768])
    _stats("lang  slice", jax_prefix_final[:, 768:], pt_prefix_final[:, 768:])


if __name__ == "__main__":
    main()
