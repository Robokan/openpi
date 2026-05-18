#!/usr/bin/env python3
"""Per-layer hidden-state comparison through the 18 gemma_expert layers.

Approach:
  - JAX: monkey-patch gemma.Block.__call__ to record the SUFFIX output of
    each scanned block via jax.experimental.io_callback. This works
    because io_callback is allowed inside JIT and runs eagerly.
  - PT : run the joint forward (PaliGemmaWithExpert.forward) and register
    forward hooks on each layer of gemma_expert.

Identical inputs (seed=42 noise, real obs, FIXED_T=0.5) so any per-layer
deviation reflects a real numerical / structural difference, not input.
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
FIXED_T = 0.5


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

    horizon = jax_policy._model.action_horizon
    adim = jax_policy._model.action_dim
    np.random.seed(42)
    noise_np = np.random.randn(1, horizon, adim).astype(np.float32)
    print(f"Fixed noise norm={np.linalg.norm(noise_np):.3f}, t={FIXED_T}")

    # ---------------------------------------------------------------
    # JAX: capture per-layer suffix outputs via io_callback hook
    # ---------------------------------------------------------------
    print("\n=== JAX per-layer capture (via disable_jit + Python list) ===")
    from openpi.models.gemma import Block as _Block
    jax_layer_outs = []
    orig_call = _Block.__call__

    def patched_call(self, xs, kv_cache, positions, attn_mask, adarms_cond_in, deterministic=True):  # noqa: FBT002
        outs, kv = orig_call(self, xs, kv_cache, positions, attn_mask, adarms_cond_in, deterministic)
        suffix_out = outs[1]
        if suffix_out is not None:
            # In disable_jit mode, this is a concrete array.
            try:
                jax_layer_outs.append(np.asarray(suffix_out).copy())
            except Exception as e:  # noqa: BLE001
                print(f"  capture skip: {e}")
        return outs, kv

    _Block.__call__ = patched_call
    try:
        # disable_jit so the patched body actually runs in Python and we
        # see concrete tensors. Scan + remat are bypassed in disable_jit.
        with jax.disable_jit():
            noise_jax = jnp.asarray(noise_np)
            t_jax = jnp.asarray(FIXED_T, dtype=jnp.float32)
            prefix_tokens, prefix_mask, prefix_ar_mask = jax_policy._model.embed_prefix(jax_obs_p)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = jax_policy._model.embed_suffix(
                jax_obs_p, noise_jax, jnp.broadcast_to(t_jax, (1,))
            )
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = jax_policy._model.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )
            v_t_jax = jax_policy._model.action_out_proj(suffix_out[:, -horizon:])
            suffix_out_jax = suffix_out
    finally:
        _Block.__call__ = orig_call

    print(f"JAX captured {len(jax_layer_outs)} layer outputs")
    if len(jax_layer_outs) == 0:
        print("ERROR: disable_jit didn't capture anything")
        return
    if len(jax_layer_outs) > 18:
        jax_layer_outs = jax_layer_outs[-18:]
    for li in [0, 8, 17]:
        if li < len(jax_layer_outs):
            print(f"  JAX layer {li}: shape={jax_layer_outs[li].shape}  |norm|={np.linalg.norm(jax_layer_outs[li]):.3f}")
    suffix_out_jax_np = np.asarray(suffix_out_jax.astype(jnp.float32))

    # ---------------------------------------------------------------
    # PT: register forward hooks on each gemma_expert layer
    # ---------------------------------------------------------------
    print("\n=== PT per-layer capture (via forward_hook) ===")
    pt_model = pt_policy._model
    pt_model.eval()
    device = "cuda"

    pt_layer_outs = []
    pt_handles = []
    expert_layers = pt_model.paligemma_with_expert.gemma_expert.model.layers

    def make_hook(layer_idx):
        def hook(mod, inputs, output):
            # output of GemmaDecoderLayer is a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            pt_layer_outs.append(h.detach().float().cpu().numpy().copy())
        return hook

    # We need to intercept the per-layer output inside compute_layer_complete.
    # But compute_layer_complete is a nested function, not a Module.
    # So we'll have to call the joint forward and hook on the per-layer
    # output by inserting our hook into the loop -- which means we need
    # to either:
    #   (a) refactor compute_layer_complete to a method, or
    #   (b) re-implement the per-layer loop here in the diag script.
    # Choosing (b) for now -- it's contained to this diag.

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = pt_model._preprocess_observation(pt_obs, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        x_t = torch.from_numpy(noise_np).to(device)
        t_pt = torch.tensor([FIXED_T], dtype=torch.float32, device=device)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pt_model.embed_suffix(state, x_t, t_pt)

        bsize = suffix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        from openpi.models_pytorch.gemma_pytorch import fp32_attention
        from transformers.models.gemma import modeling_gemma

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        suffix_to_prefix_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        prefix_to_suffix_masks = torch.zeros(bsize, prefix_len, suffix_len, dtype=torch.bool, device=device)
        top_row = torch.cat([prefix_att_2d_masks, prefix_to_suffix_masks], dim=2)
        bottom_row = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks = torch.cat([top_row, bottom_row], dim=1)
        full_att_2d_masks_4d = pt_model._prepare_attention_masks_4d(full_att_2d_masks)

        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

        # bf16 cast
        pg_lm = pt_model.paligemma_with_expert.paligemma.language_model
        expert_m = pt_model.paligemma_with_expert.gemma_expert.model
        if pg_lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)
        if expert_m.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        adarms_cond_list = [None, adarms_cond]

        prefix_h = prefix_embs
        suffix_h = suffix_embs
        models = [pg_lm, expert_m]

        num_layers = pg_lm.config.num_hidden_layers
        for layer_idx in range(num_layers):
            inputs_embeds = [prefix_h, suffix_h]
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]
                hs_normed, gate = layer.input_layernorm(hidden_states, cond=adarms_cond_list[i])
                gates.append(gate)
                input_shape = hs_normed.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                q = layer.self_attn.q_proj(hs_normed).view(hidden_shape).transpose(1, 2)
                k = layer.self_attn.k_proj(hs_normed).view(hidden_shape).transpose(1, 2)
                v = layer.self_attn.v_proj(hs_normed).view(hidden_shape).transpose(1, 2)
                query_states.append(q); key_states.append(k); value_states.append(v)

            q_cat = torch.cat(query_states, dim=2)
            k_cat = torch.cat(key_states, dim=2)
            v_cat = torch.cat(value_states, dim=2)

            dummy = torch.zeros(q_cat.shape[0], q_cat.shape[2], q_cat.shape[-1], device=device, dtype=q_cat.dtype)
            cos, sin = pg_lm.rotary_emb(dummy, position_ids)
            q_cat, k_cat = modeling_gemma.apply_rotary_pos_emb(q_cat, k_cat, cos, sin, unsqueeze_dim=1)

            self_attn = pg_lm.layers[layer_idx].self_attn
            scaling = self_attn.scaling
            num_kv_groups = self_attn.num_key_value_groups
            seq_len = k_cat.shape[2]
            head_dim_kv = k_cat.shape[-1]
            B = q_cat.shape[0]
            k_exp = k_cat[:, :, None, :, :].expand(B, k_cat.shape[1], num_kv_groups, seq_len, head_dim_kv).reshape(B, -1, seq_len, head_dim_kv)
            v_exp = v_cat[:, :, None, :, :].expand(B, v_cat.shape[1], num_kv_groups, seq_len, head_dim_kv).reshape(B, -1, seq_len, head_dim_kv)
            att_output = fp32_attention(q_cat, k_exp, v_exp, additive_mask_4d=full_att_2d_masks_4d, scaling=scaling)
            att_output = att_output.transpose(1, 2).contiguous()
            head_dim = self_attn.head_dim
            att_output = att_output.reshape(B, -1, 1 * 8 * head_dim)

            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]
                ao = att_output[:, start_pos:end_pos]
                if ao.dtype != layer.self_attn.o_proj.weight.dtype:
                    ao = ao.to(layer.self_attn.o_proj.weight.dtype)
                oe = layer.self_attn.o_proj(ao)
                oe = modeling_gemma._gated_residual(hidden_states, oe, gates[i])
                after_first_residual = oe.clone()
                oe, gate2 = layer.post_attention_layernorm(oe, cond=adarms_cond_list[i])
                if hasattr(layer.mlp, "up_proj") and layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    oe = oe.to(torch.bfloat16)
                mlp_out = layer.mlp(oe)
                final = modeling_gemma._gated_residual(after_first_residual, mlp_out, gate2)
                outputs_embeds.append(final)
                start_pos = end_pos

            prefix_h, suffix_h = outputs_embeds
            pt_layer_outs.append(suffix_h.detach().float().cpu().numpy().copy())

    print(f"PT  captured {len(pt_layer_outs)} layer outputs")

    # ---------------------------------------------------------------
    # Compare per layer
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PER-LAYER GEMMA_EXPERT SUFFIX HIDDEN STATE COMPARISON (T=0.5, seed=42)")
    print("=" * 100)
    print(f"{'layer':>5s}  {'cos':>10s}  {'|jax|':>10s}  {'|pt|':>10s}  {'ratio':>8s}  {'max|d|':>10s}")
    for li in range(min(len(jax_layer_outs), len(pt_layer_outs))):
        j = jax_layer_outs[li].astype(np.float32)
        p = pt_layer_outs[li].astype(np.float32)
        if j.shape != p.shape:
            print(f"  {li:>5d}  shape mismatch: jax={j.shape} pt={p.shape}")
            continue
        jf, pf = j.flatten(), p.flatten()
        cos = float((jf @ pf) / (np.linalg.norm(jf) * np.linalg.norm(pf) + 1e-12))
        nj, npt = float(np.linalg.norm(j)), float(np.linalg.norm(p))
        print(f"  {li:>5d}  {cos:>+10.6f}  {nj:>10.3f}  {npt:>10.3f}  {npt/(nj+1e-12):>8.4f}  {float(np.max(np.abs(j-p))):>10.4f}")


if __name__ == "__main__":
    main()
