#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # Convert to PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import json
import os
import pathlib
import shutil
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def _extract_lora_pt(
    flat_params: dict,
    *,
    paligemma_num_layers: int = 18,
    paligemma_num_heads: int = 8,
    paligemma_head_dim: int = 256,
    paligemma_scaling: float = 1.0,
    expert_num_layers: int = 18,
    expert_num_heads: int = 8,
    expert_head_dim: int = 256,
    expert_scaling: float = 1.0,
) -> tuple[dict, dict]:
    """Extract JAX LoRA tensors into PT-named runtime-LoRA form.

    Returns:
      (flat_params_without_lora, pt_lora_dict)

    `pt_lora_dict` keys follow the convention:
      "paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.lora_a"
      "...q_proj.lora_b"
      "...q_proj.lora_scaling"  (scalar tensor)
    and similar for k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj on both
    paligemma and gemma_expert subtrees.

    Tensor shapes are documented in `src/openpi/models_pytorch/lora_runtime.py`.
    For o_proj specifically we pre-sum lora_b across the N (head) axis in fp32
    so the runtime can do a single (B,T,L) @ (L,D) matmul — this matches
    JAX's einsum semantics where the unmatched N axis is implicitly summed.
    """
    f = dict(flat_params)
    pt_lora: dict = {}

    def _pop(k):
        if k in f:
            return f.pop(k)
        return None

    def _add(pt_path: str, la: np.ndarray, lb: np.ndarray, scaling: float):
        pt_lora[f"{pt_path}.lora_a"] = la.astype(np.float32)
        pt_lora[f"{pt_path}.lora_b"] = lb.astype(np.float32)
        pt_lora[f"{pt_path}.lora_scaling"] = np.float32(scaling)

    has_paligemma_lora = "llm/layers/attn/q_einsum/lora_a" in f
    has_expert_lora = "llm/layers/attn/q_einsum_1/lora_a" in f

    if has_paligemma_lora:
        q_la = _pop("llm/layers/attn/q_einsum/lora_a")  # (L, N, D, R)
        q_lb = _pop("llm/layers/attn/q_einsum/lora_b")  # (L, N, R, H)
        kv_la = _pop("llm/layers/attn/kv_einsum/lora_a")  # (L, 2, 1, D, R)
        kv_lb = _pop("llm/layers/attn/kv_einsum/lora_b")  # (L, 2, 1, R, H)
        o_la = _pop("llm/layers/attn/attn_vec_einsum/lora_a")  # (L, N, H, R)
        o_lb = _pop("llm/layers/attn/attn_vec_einsum/lora_b")  # (L, N, R, D)
        mlp_g_la = _pop("llm/layers/mlp/gating_einsum_lora_a")  # (L, 2, D, R)
        mlp_g_lb = _pop("llm/layers/mlp/gating_einsum_lora_b")  # (L, 2, R, I)
        mlp_l_la = _pop("llm/layers/mlp/linear_lora_a")  # (L, I, R)
        mlp_l_lb = _pop("llm/layers/mlp/linear_lora_b")  # (L, R, D)

        s = paligemma_scaling
        pg_base = "paligemma_with_expert.paligemma.model.language_model.layers"
        for i in range(paligemma_num_layers):
            _add(f"{pg_base}.{i}.self_attn.q_proj", q_la[i], q_lb[i], s)  # (N, D, R), (N, R, H)
            _add(f"{pg_base}.{i}.self_attn.k_proj", kv_la[i, 0, 0], kv_lb[i, 0, 0], s)  # (D, R), (R, H)
            _add(f"{pg_base}.{i}.self_attn.v_proj", kv_la[i, 1, 0], kv_lb[i, 1, 0], s)
            # o_proj: la (N, H, R) -> (N*H, R); lb (N, R, D) -> (R, D) via sum over N
            o_la_flat = o_la[i].reshape(-1, o_la[i].shape[-1])  # (N*H, R)
            o_lb_summed = o_lb[i].sum(axis=0)  # (R, D)
            _add(f"{pg_base}.{i}.self_attn.o_proj", o_la_flat, o_lb_summed, s)
            _add(f"{pg_base}.{i}.mlp.gate_proj", mlp_g_la[i, 0], mlp_g_lb[i, 0], s)  # (D, R), (R, I)
            _add(f"{pg_base}.{i}.mlp.up_proj", mlp_g_la[i, 1], mlp_g_lb[i, 1], s)
            _add(f"{pg_base}.{i}.mlp.down_proj", mlp_l_la[i], mlp_l_lb[i], s)  # (I, R), (R, D)

    if has_expert_lora:
        q_la = _pop("llm/layers/attn/q_einsum_1/lora_a")
        q_lb = _pop("llm/layers/attn/q_einsum_1/lora_b")
        kv_la = _pop("llm/layers/attn/kv_einsum_1/lora_a")
        kv_lb = _pop("llm/layers/attn/kv_einsum_1/lora_b")
        o_la = _pop("llm/layers/attn/attn_vec_einsum_1/lora_a")
        o_lb = _pop("llm/layers/attn/attn_vec_einsum_1/lora_b")
        mlp_g_la = _pop("llm/layers/mlp_1/gating_einsum_lora_a")
        mlp_g_lb = _pop("llm/layers/mlp_1/gating_einsum_lora_b")
        mlp_l_la = _pop("llm/layers/mlp_1/linear_lora_a")
        mlp_l_lb = _pop("llm/layers/mlp_1/linear_lora_b")

        s = expert_scaling
        ex_base = "paligemma_with_expert.gemma_expert.model.layers"
        for i in range(expert_num_layers):
            _add(f"{ex_base}.{i}.self_attn.q_proj", q_la[i], q_lb[i], s)
            _add(f"{ex_base}.{i}.self_attn.k_proj", kv_la[i, 0, 0], kv_lb[i, 0, 0], s)
            _add(f"{ex_base}.{i}.self_attn.v_proj", kv_la[i, 1, 0], kv_lb[i, 1, 0], s)
            o_la_flat = o_la[i].reshape(-1, o_la[i].shape[-1])
            o_lb_summed = o_lb[i].sum(axis=0)
            _add(f"{ex_base}.{i}.self_attn.o_proj", o_la_flat, o_lb_summed, s)
            _add(f"{ex_base}.{i}.mlp.gate_proj", mlp_g_la[i, 0], mlp_g_lb[i, 0], s)
            _add(f"{ex_base}.{i}.mlp.up_proj", mlp_g_la[i, 1], mlp_g_lb[i, 1], s)
            _add(f"{ex_base}.{i}.mlp.down_proj", mlp_l_la[i], mlp_l_lb[i], s)

    print(f"  Extracted {len(pt_lora)} LoRA tensors (paligemma={has_paligemma_lora}, expert={has_expert_lora})")
    return f, pt_lora


def _merge_lora_into_base(flat_params: dict, scaling: float = 1.0) -> dict:
    """Merge LoRA delta weights into their base weight tensors and drop the LoRA keys.

    JAX checkpoint stores LoRA as separate `lora_a` / `lora_b` matrices alongside the
    base weight `w` (or alongside a same-named base parameter for FeedForward layers).
    The PyTorch model has no LoRA layers, so we must fold the deltas into the base
    weight before exporting. With `alpha == rank` (which is the case for pi05 LoRA
    configs: gemma_2b_lora rank=16/alpha=16, gemma_300m_lora rank=32/alpha=32),
    the per-layer scaling is exactly 1.0.

    Mutates and returns a NEW dict so the caller can decide what to do with it.
    """
    merged = dict(flat_params)
    consumed = set()
    pairs = []  # (lora_a_key, lora_b_key, base_key)

    for k in list(merged.keys()):
        if k.endswith("/lora_a"):
            stem = k[: -len("/lora_a")]
            pairs.append((k, stem + "/lora_b", stem + "/w"))
        elif k.endswith("_lora_a"):
            # FeedForward style: gating_einsum_lora_a sits next to gating_einsum
            stem = k[: -len("_lora_a")]
            pairs.append((k, stem + "_lora_b", stem))

    if not pairs:
        return merged

    print(f"  Found {len(pairs)} LoRA weight pairs to merge into base weights")
    for la_k, lb_k, base_k in pairs:
        if base_k not in merged:
            raise KeyError(f"LoRA base weight not found: {base_k} (referenced by {la_k})")
        if lb_k not in merged:
            raise KeyError(f"LoRA pair missing: {lb_k} for {la_k}")

        w = merged[base_k]
        la = merged[la_k]
        lb = merged[lb_k]

        # Sanity-check shapes: la is [..., in, rank], lb is [..., rank, out], w is [..., in, out]
        if la.shape[:-2] != lb.shape[:-2] or la.shape[:-2] != w.shape[:-2]:
            raise ValueError(
                f"LoRA leading-dim mismatch for {base_k}: w={w.shape} la={la.shape} lb={lb.shape}"
            )
        if la.shape[-2] != w.shape[-2] or lb.shape[-1] != w.shape[-1] or la.shape[-1] != lb.shape[-2]:
            raise ValueError(
                f"LoRA inner-dim mismatch for {base_k}: w={w.shape} la={la.shape} lb={lb.shape}"
            )

        # matmul broadcasts over leading dims; treats last two dims as the matrix
        delta = np.matmul(la.astype(np.float32), lb.astype(np.float32))
        merged[base_k] = (w.astype(np.float32) + scaling * delta).astype(w.dtype)
        consumed.add(la_k)
        consumed.add(lb_k)

    for k in consumed:
        del merged[k]

    print(f"  Merged LoRA into {len(pairs)} base weights ({len(consumed)} LoRA tensors dropped)")
    return merged


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    paligemma_flat = traversals.flatten_mapping(params["PaliGemma"], sep="/")
    # `OPENPI_PT_RUNTIME_LORA=1` keeps LoRA tensors separate from base weights
    # for runtime-LoRA application in PyTorch (matches JAX two-matmul order
    # exactly, no bf16 merge-precision loss). Default behavior pre-merges
    # LoRA into base weights for backwards compatibility.
    # `OPENPI_CONV_LORA_SCALING` overrides scaling for ablation (e.g. set to 0
    # to produce a base-only PT model with no LoRA contribution).
    import os as _os  # noqa: PLC0415
    _scaling = float(_os.environ.get("OPENPI_CONV_LORA_SCALING", "1.0"))
    _runtime_lora = _os.environ.get("OPENPI_PT_RUNTIME_LORA", "0").lower() in ("1", "true", "yes")
    pt_lora: dict = {}
    if _runtime_lora:
        paligemma_flat, pt_lora = _extract_lora_pt(paligemma_flat, paligemma_scaling=_scaling, expert_scaling=_scaling)
        print(f"  RUNTIME LoRA mode: kept LoRA separate ({len(pt_lora)} PT tensors), base weights left unmerged")
    else:
        paligemma_flat = _merge_lora_into_base(paligemma_flat, scaling=_scaling)
    return {"paligemma_params": paligemma_flat, "projection_params": params, "pt_lora": pt_lora}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # Process projection params
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # Instantiate model. NOTE: PI0Pytorch.__init__ -> PaliGemmaWithExpertModel
    # internally calls `self.to_bfloat16_for_selected_params(config.dtype)` which
    # downcasts the freshly-initialized weights to bf16 (when config.dtype="bfloat16").
    # Then a plain `load_state_dict(strict=False)` would *cast our fp32 LoRA-merged
    # weights down to bf16 to match the model's parameter dtype*, losing precision
    # in the small LoRA delta. That bf16-truncation is exactly the source of the
    # observed ~1% per-weight LoRA mismatch that compounds to ~8% post-unnormalize
    # action magnitude bias on the OpenArm robot.
    #
    # FIX: instantiate the model with `dtype='float32'` so its parameters start
    # in fp32. Then load_state_dict (default cast-to-dest-dtype) is a no-op for
    # dtype, preserving the fp32 precision of the LoRA-merged weights. We can
    # then optionally cast the WHOLE model to bf16 just before save (if the
    # user requested precision='bfloat16'). The runtime cast in policy_config
    # already handles the bf16 inference path with the correct `assign=True`
    # mechanism, so saving fp32 is the safer canonical form.
    #
    # Crucially this avoids `assign=True` here, which breaks the lm_head <->
    # embed_tokens weight tying (HF PaliGemma) and produces broken inference.
    import dataclasses as _dc
    fp32_model_config = _dc.replace(model_config, dtype="float32")
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(fp32_model_config)

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Standard load_state_dict (NOT assign=True). Since the model is fp32, the
    # implicit cast-to-destination is fp32->fp32 = no-op. The fp32 LoRA-merged
    # weights are preserved exactly.
    pi0_model.load_state_dict(all_params, strict=False)

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # If runtime-LoRA mode is active, also save the LoRA tensors separately
    # alongside the base safetensors. They are loaded at inference time by
    # `lora_runtime.install_runtime_lora` which patches the projection forwards.
    pt_lora = initial_params.get("pt_lora", {})
    if pt_lora:
        lora_tensors = {}
        for k, v in pt_lora.items():
            t = torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else torch.as_tensor(v)
            if precision == "bfloat16" and t.dtype == torch.float32:
                t = t.to(torch.bfloat16)
            lora_tensors[k] = t
        lora_path = os.path.join(output_path, "lora.safetensors")
        safetensors.torch.save_file(lora_tensors, lora_path)
        print(f"  Saved {len(lora_tensors)} runtime-LoRA tensors to {lora_path}")

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion
        inspect_only: Only inspect parameter keys, don't convert
    """
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
