import os
from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


def fp32_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   additive_mask_4d: torch.Tensor, scaling: float) -> torch.Tensor:
    """Attention with fp32 logits + softmax, matching JAX's
    `jnp.einsum(..., preferred_element_type=jnp.float32)` semantics.

    PyTorch's `scaled_dot_product_attention` and `eager_attention_forward` keep
    Q @ K^T in the input dtype (bf16 at inference). With ~1000-token prefix
    sequences the bf16 accumulator drifts significantly, primarily on language
    tokens at high positions (cos ~0.65 vs JAX). Computing the matmul + softmax
    in fp32 (and downcasting probs before the V matmul) recovers near-perfect
    parity with the JAX prefix forward.

    Args:
        query: (B, H, T, D) in any float dtype.
        key:   (B, H, S, D) in any float dtype (post-GQA repeat).
        value: (B, H, S, D) in any float dtype.
        additive_mask_4d: (B, 1, T, S) float mask with 0 or -big_neg.
        scaling: scalar attention scale (typically 1/sqrt(head_dim)).

    Returns:
        attention output, shape (B, H, T, D), in the input query.dtype.
    """
    orig_dtype = query.dtype
    q_f32 = query.float()
    k_f32 = key.float()
    v_f32 = value.float()
    logits = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * scaling
    logits = logits + additive_mask_4d.float()
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v_f32)
    return out.to(orig_dtype)


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

        # CRITICAL FIX: `self.to(dtype=torch.bfloat16)` above casts BUFFERS as
        # well as parameters. The `inv_freq` buffer of every `GemmaRotaryEmbedding`
        # gets truncated from fp32 to bf16, losing ~3 decimal digits of precision
        # (e.g. inv_freq[1] 0.9305720 -> 0.9296875). This compounds through cos/sin
        # at every position and propagates through all 18 layers x 10 denoise
        # steps, producing a ~8% chunk-level magnitude bias on PT actions vs JAX
        # (joint 3 +0.135 rad / "arms drift upward" on OpenArm).
        #
        # We restore inv_freq to fp32 here. JAX computes RoPE in fp32 throughout
        # (`_apply_rope` line 426: `dtype=jnp.float32`), so this just matches JAX.
        for name, buf in self.named_buffers():
            if name.endswith("rotary_emb.inv_freq") and buf.dtype != torch.float32:
                # Re-attach as fp32. We can't easily compute fresh inv_freq here
                # without knowing the module's config, so we upcast the existing
                # buffer and accept the bf16->fp32 promotion (still better than
                # bf16 throughout). Then we also re-initialize via the original
                # rope_init_fn for any RotaryEmbedding modules we can find.
                buf.data = buf.data.to(dtype=torch.float32)

        # Re-initialize inv_freq from the config to recover full fp32 precision
        # (the .to(fp32) above only widens the dtype but the values were already
        # truncated to bf16-representable). We do this by walking submodules and
        # re-running their rope_init_fn.
        for module in self.modules():
            if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq"):
                try:
                    new_inv_freq, _ = module.rope_init_fn(module.config, device=module.inv_freq.device)
                    module.inv_freq.data = new_inv_freq.to(dtype=torch.float32)
                except Exception:  # noqa: BLE001
                    pass

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                self_attn = self.paligemma.language_model.layers[layer_idx].self_attn
                scaling = self_attn.scaling

                # JAX uses fp32-accumulator einsum (preferred_element_type=jnp.float32)
                # for attention logits. PT's eager attention stays in bf16, which drops
                # language-token cos to ~0.65 vs JAX over ~1000 prefix keys. Use fp32
                # attention to recover parity.
                # K, V are not yet GQA-expanded; expand here (matches eager_attention_forward).
                num_kv_groups = self_attn.num_key_value_groups
                seq_len = key_states.shape[2]
                head_dim_kv = key_states.shape[-1]
                k_expanded = (
                    key_states[:, :, None, :, :]
                    .expand(batch_size, key_states.shape[1], num_kv_groups, seq_len, head_dim_kv)
                    .reshape(batch_size, -1, seq_len, head_dim_kv)
                )
                v_expanded = (
                    value_states[:, :, None, :, :]
                    .expand(batch_size, value_states.shape[1], num_kv_groups, seq_len, head_dim_kv)
                    .reshape(batch_size, -1, seq_len, head_dim_kv)
                )
                if os.environ.get("OPENPI_PT_FP32_ATTN", "1") != "0":
                    att_output = fp32_attention(
                        query_states, k_expanded, v_expanded,
                        additive_mask_4d=attention_mask,
                        scaling=scaling,
                    )
                else:
                    # Default bf16 SDPA-style attention (no fp32 logit accumulator).
                    # Comparison anchor for OPENPI_PT_FP32_ATTN ablation.
                    _logits = torch.matmul(query_states, k_expanded.transpose(-1, -2)) * scaling
                    _logits = _logits + attention_mask
                    _probs = torch.softmax(_logits, dim=-1)
                    att_output = torch.matmul(_probs, v_expanded)
                # att_output: (B, num_heads, T, head_dim) -> (B, T, num_heads*head_dim)
                att_output = att_output.transpose(1, 2).contiguous()
                head_dim = self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
