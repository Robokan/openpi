#!/usr/bin/env python3
"""Compare a specific layer's weights (post-LoRA-merge) JAX vs PT to find conversion bugs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")

CKPT_JAX = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
CKPT_PT = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"


def _stat(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    cos = float((a.flatten() @ b.flatten()) / (np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-12))
    print(f"  [{name}] cos={cos:+.6f}  max|diff|={diff.max():.6f}  mean|diff|={diff.mean():.6f}  |a|_mean={np.abs(a).mean():.4f}  |b|_mean={np.abs(b).mean():.4f}")


def main() -> int:
    import safetensors.torch as st
    import importlib.util
    from openpi.models import model as _model
    from flax import traverse_util as traversals

    spec = importlib.util.spec_from_file_location("convert_mod", "/app/examples/convert_jax_model_to_pytorch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _merge_lora_into_base = mod._merge_lora_into_base

    print("Loading JAX params + merging LoRA ...")
    params = _model.restore_params(f"{CKPT_JAX}/params/", restore_type=np.ndarray, dtype=None)
    paligemma_flat = traversals.flatten_dict(params["PaliGemma"], sep="/")
    paligemma_flat = _merge_lora_into_base(paligemma_flat, scaling=1.0)

    pt_state = st.load_file(str(Path(CKPT_PT) / "model.safetensors"))

    # Try matching paligemma q_proj layer 0:
    # JAX: 'llm/layers/attn/q_einsum/w' shape (18, 8, 2048, 256) per layer
    # PT:  'paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight' shape (2048, 2048)
    layer_idx = 0
    print(f"\n=== Paligemma q_proj layer {layer_idx} ===")
    jq = np.asarray(paligemma_flat["llm/layers/attn/q_einsum/w"])[layer_idx]  # (8, 2048, 256)
    print(f"  JAX shape: {jq.shape}")
    pq = pt_state[f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"].float().numpy()
    print(f"  PT  shape: {pq.shape}")
    # Convert JAX (heads, in, head_dim) -> (heads*head_dim, in)
    jq_pt_layout = jq.transpose(0, 2, 1).reshape(jq.shape[0] * jq.shape[2], jq.shape[1])
    print(f"  JAX -> PT layout: {jq_pt_layout.shape}")
    _stat("paligemma q_proj L0", jq_pt_layout, pq)

    # Expert q_proj layer 0:
    print(f"\n=== Expert q_proj layer {layer_idx} ===")
    jq = np.asarray(paligemma_flat["llm/layers/attn/q_einsum_1/w"])[layer_idx]  # (8, 1024, 256)
    print(f"  JAX shape: {jq.shape}")
    pq = pt_state[f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.self_attn.q_proj.weight"].float().numpy()
    print(f"  PT  shape: {pq.shape}")
    jq_pt_layout = jq.transpose(0, 2, 1).reshape(jq.shape[0] * jq.shape[2], jq.shape[1])
    _stat("expert q_proj L0", jq_pt_layout, pq)

    # Expert input_layernorm.dense.weight (adaRMS)
    print(f"\n=== Expert input_layernorm.dense L0 ===")
    j_dense = np.asarray(paligemma_flat["llm/layers/pre_attention_norm_1/Dense_0/kernel"])[layer_idx]  # (1024, 3072)
    print(f"  JAX shape: {j_dense.shape}")
    p_dense = pt_state[f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.input_layernorm.dense.weight"].float().numpy()
    print(f"  PT  shape: {p_dense.shape}")
    _stat("dense L0 (transpose)", j_dense.T, p_dense)

    # Expert input_layernorm.dense.bias
    print(f"\n=== Expert input_layernorm.dense.bias L0 ===")
    j_b = np.asarray(paligemma_flat["llm/layers/pre_attention_norm_1/Dense_0/bias"])[layer_idx]
    p_b = pt_state[f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.input_layernorm.dense.bias"].float().numpy()
    print(f"  JAX shape: {j_b.shape}  PT shape: {p_b.shape}")
    _stat("dense.bias L0", j_b, p_b)

    # MLP gating_einsum (paligemma)
    print(f"\n=== Paligemma mlp gating L0 ===")
    j_gate = np.asarray(paligemma_flat["llm/layers/mlp/gating_einsum"])[layer_idx]  # (2, 2048, 16384)
    p_gate = pt_state[f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"].float().numpy()
    print(f"  JAX shape: {j_gate.shape}  PT gate_proj shape: {p_gate.shape}")
    # JAX gating_einsum has shape (2, in, hidden) where dim 0 is [gate, up]
    # PT splits into gate_proj and up_proj
    _stat("gate_proj (jax[0].T)", j_gate[0].T, p_gate)
    p_up = pt_state[f"paligemma_with_expert.paligemma.model.language_model.layers.{layer_idx}.mlp.up_proj.weight"].float().numpy()
    _stat("up_proj   (jax[1].T)", j_gate[1].T, p_up)

    # Expert MLP gating_einsum
    print(f"\n=== Expert mlp gating L0 ===")
    j_gate = np.asarray(paligemma_flat["llm/layers/mlp_1/gating_einsum"])[layer_idx]
    p_gate = pt_state[f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.mlp.gate_proj.weight"].float().numpy()
    print(f"  JAX shape: {j_gate.shape}  PT gate_proj shape: {p_gate.shape}")
    _stat("gate_proj (jax[0].T)", j_gate[0].T, p_gate)
    p_up = pt_state[f"paligemma_with_expert.gemma_expert.model.layers.{layer_idx}.mlp.up_proj.weight"].float().numpy()
    _stat("up_proj   (jax[1].T)", j_gate[1].T, p_up)

    # AdaRMS dense weights (gemma_expert layers)
    print("\n=== AdaRMS dense weights (gemma_expert) ===")
    if "PaliGemma" in params and "llm" in params["PaliGemma"]:
        llm_params = params["PaliGemma"]["llm"]
        # Find AdaRMS pre_attention_norm_1 dense kernel/bias for layer 0
        for jax_key, pt_prefix in [
            ("pre_attention_norm_1", "paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm"),
            ("pre_ffw_norm_1", "paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm"),
        ]:
            try:
                kernel_arr = np.asarray(llm_params["layers"][jax_key]["Dense_0"]["kernel"])
                bias_arr = np.asarray(llm_params["layers"][jax_key]["Dense_0"]["bias"])
            except KeyError:
                print(f"  JAX has no {jax_key}/Dense_0, skip")
                continue
            for i in [0, 5, 17]:  # spot-check layers
                if i >= kernel_arr.shape[0]:
                    continue
                pt_key_w = pt_prefix.format(i=i) + ".dense.weight"
                pt_key_b = pt_prefix.format(i=i) + ".dense.bias"
                if pt_key_w not in pt_state or pt_key_b not in pt_state:
                    print(f"  PT key missing: {pt_key_w}")
                    continue
                j_kernel = kernel_arr[i]  # (in_dim, 3*dim)
                j_bias = bias_arr[i]      # (3*dim,)
                p_w = pt_state[pt_key_w].float().numpy()  # (3*dim, in_dim) — PT nn.Linear
                p_b = pt_state[pt_key_b].float().numpy()  # (3*dim,)
                # JAX (in, out) vs PT (out, in)
                _stat(f"{jax_key}/layer{i} kernel", j_kernel.T, p_w)
                _stat(f"{jax_key}/layer{i} bias",   j_bias,     p_b)
        # final norm
        try:
            final_k = np.asarray(llm_params["final_norm_1"]["Dense_0"]["kernel"])
            final_b = np.asarray(llm_params["final_norm_1"]["Dense_0"]["bias"])
        except KeyError:
            print("  No final_norm_1/Dense_0 in JAX, skip")
        else:
            pt_k = pt_state.get("paligemma_with_expert.gemma_expert.model.norm.dense.weight")
            pt_b = pt_state.get("paligemma_with_expert.gemma_expert.model.norm.dense.bias")
            if pt_k is not None and pt_b is not None:
                _stat("final_norm_1 kernel", final_k.T, pt_k.float().numpy())
                _stat("final_norm_1 bias",   final_b,   pt_b.float().numpy())

    # action_in_proj and action_out_proj from JAX top-level
    print("\n=== action_in_proj / action_out_proj / time_mlp ===")
    print("JAX top-level params keys:", list(params.keys()))
    for top in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
        if top not in params:
            print(f"  {top} not in JAX params, skip")
            continue
        # Flatten the top-level dict for this projection
        sub = params[top]
        for sub_key, sub_val in sub.items():
            jv = np.asarray(sub_val)
            # Map to PT key
            if sub_key == "kernel":
                pt_key = f"{top}.weight"
            elif sub_key == "bias":
                pt_key = f"{top}.bias"
            else:
                pt_key = None
            print(f"  JAX [{top}/{sub_key}]: shape={jv.shape}  PT key={pt_key}")
            if pt_key is None or pt_key not in pt_state:
                print(f"    PT key missing!")
                continue
            pv = pt_state[pt_key].float().numpy()
            # If 2D kernel, JAX (in, out) vs PT (out, in) -> transpose
            if sub_key == "kernel" and jv.ndim == 2:
                jv = jv.T
            _stat(f"{top}/{sub_key}", jv, pv)

    # Verify language embedding table:
    print(f"\n=== Searching for embedding key in safetensors ===")
    embed_keys = [k for k in pt_state if "embed" in k.lower()]
    for k in embed_keys:
        print(f"  {k}  shape={tuple(pt_state[k].shape)}")

    # Load the actual PT model and inspect its embed_tokens.weight
    print(f"\n=== Loading PT model and inspecting embed_tokens.weight ===")
    import torch
    from openpi.models_pytorch import pi0_pytorch as _pi0_pt
    from openpi.training import config as _config
    cfg = _config.get_config("pi05_openarm_ngc_lora_v4")
    pt_model = _pi0_pt.PI0Pytorch(config=cfg.model)
    pre_load_emb = pt_model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.detach().clone()
    print(f"  pre-load embed_tokens.weight  shape={tuple(pre_load_emb.shape)}  |mean|={pre_load_emb.abs().mean().item():.6f}")

    miss, unexp = pt_model.load_state_dict(pt_state, strict=False, assign=True)
    post_load_emb = pt_model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.detach().clone()
    print(f"  post-load embed_tokens.weight shape={tuple(post_load_emb.shape)}  |mean|={post_load_emb.abs().mean().item():.6f}")
    print(f"  missing keys count: {len(miss)}")
    print(f"  is embed_tokens in missing? {('paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight' in miss)}")
    # Compare to JAX
    j_emb = np.asarray(paligemma_flat["llm/embedder/input_embedding"])
    print(f"\n  JAX  embedding |mean|={np.abs(j_emb).mean():.6f}")
    print(f"  PT   embedding |mean|={post_load_emb.abs().mean().item():.6f}")
    print(f"  JAX  embed[7071] norm = {np.linalg.norm(j_emb[7071]):.4f}")
    print(f"  PT   embed[7071] norm = {post_load_emb[7071].float().norm().item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
