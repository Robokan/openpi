"""Run ONE denoise step (NOT a full diffusion loop) on both JAX and PT
with identical observation + identical noise + identical fixed time, and
compare the per-step velocity prediction v_t.

This isolates the model's per-step compute error from any compounding through
the diffusion integration.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")

import numpy as np
import jax
import jax.numpy as jnp
import torch

from openpi.training import config as _config
from openpi.policies import policy_config as _pc
from openpi.models import model as _model
from openpi.shared import download as _dl


CONFIG_NAME = "pi05_openarm_ngc_lora_v4"
JAX_DIR = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999"
PT_DIR = "/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_pytorch"
FIXED_TIME = 0.5


def load_jax_model():
    cfg = _config.get_config(CONFIG_NAME)
    jax_path = _dl.maybe_download(JAX_DIR + "/params")
    model = cfg.model.load(_model.restore_params(jax_path, dtype=jnp.bfloat16))
    return cfg, model


def load_pt_policy():
    cfg = _config.get_config(CONFIG_NAME)
    policy = _pc.create_trained_policy(cfg, PT_DIR, pytorch_device="cuda")
    return cfg, policy


def build_synthetic_observation():
    """Single deterministic synthetic observation: zeros for state + prompt."""
    return {
        "state": np.zeros((16,), dtype=np.float32),
        "images": {
            "cam_high": (np.ones((224, 224, 3)) * 127).astype(np.uint8),
            "cam_left_wrist": (np.ones((224, 224, 3)) * 127).astype(np.uint8),
            "cam_right_wrist": (np.ones((224, 224, 3)) * 127).astype(np.uint8),
        },
        "prompt": "put the chocolate bars in the container",
    }


def to_jax_obs(obs_pre, model):
    """Run JAX input transforms (norm, prompt, resize) and convert to Observation."""
    import openpi.transforms as _T
    from openpi.policies import openarm_policy

    # Apply same transforms as the policy server: input transforms
    transforms = openarm_policy.OpenArmInputs(
        action_dim=model.action_dim,
        model_type=model.model_type,
    )
    norm_path = _dl.maybe_download(PT_DIR + "/assets/openarm/norm_stats.json")
    norm_stats = _T.deserialize_json(norm_path.read_text())["norm_stats"]
    pipeline = _T.compose(
        [transforms, _T.ResizeImages(224, 224), _T.Normalize(norm_stats, use_quantiles=False), _T.TokenizePrompt(_T.PaligemmaTokenizer(model.max_token_len))]
    )
    obs = pipeline(obs_pre)
    return obs


def main():
    print("=== Loading JAX model ===")
    jax_cfg, jax_model = load_jax_model()
    print(f"JAX loaded. action_dim={jax_model.action_dim}  horizon={jax_model.action_horizon}")

    print("\n=== Loading PT policy ===")
    pt_cfg, pt_policy = load_pt_policy()
    pt_model = pt_policy._model
    device = next(pt_model.parameters()).device
    print(f"PT loaded. device={device}")

    print("\n=== Build synthetic observation + transform via PT policy ===")
    obs_raw = build_synthetic_observation()
    # Use the PT policy's input transforms to get a normalized + tokenized obs
    obs_pre = pt_policy._input_transform(obs_raw)
    print(f"Transformed obs keys: {list(obs_pre.keys())}")

    # Build JAX Observation from same obs_pre
    jax_obs = _model.Observation.from_dict(obs_pre)
    pt_obs = _model.Observation.from_dict({k: (torch.from_numpy(np.asarray(v)[None]).to(device) if isinstance(v, np.ndarray) else v) for k, v in obs_pre.items()})
    # Better: convert jax_obs to PT obs by adding batch dim
    # Actually let's just rebuild PT-style obs from obs_pre

    print("\n=== Fixed noise + fixed time ===")
    horizon = jax_model.action_horizon
    adim = jax_model.action_dim
    np.random.seed(0)
    noise_np = np.random.randn(horizon, adim).astype(np.float32)
    noise_jax = jnp.asarray(noise_np)[None]  # (1, horizon, adim)
    noise_pt = torch.from_numpy(noise_np)[None].to(device)
    time_val = FIXED_TIME
    time_jax = jnp.array(time_val, dtype=jnp.float32)
    time_pt = torch.tensor([time_val], dtype=torch.float32, device=device)

    print("\n=== JAX: single-step forward ===")
    # Call internal embed_prefix + embed_suffix + llm to get v_t at t=FIXED_TIME with noise=noise_np
    @jax.jit
    def jax_one_step(rng, obs, noise, time):
        observation = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = jax_model.embed_prefix(observation)
        import einops
        prefix_attn_mask = _model.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = jax_model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        x_t = noise
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = jax_model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, observation.state.shape[0])
        )
        suffix_attn_mask = _model.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = jax_model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = jax_model.action_out_proj(suffix_out[:, -horizon:])
        return v_t, suffix_out[:, -horizon:]

    v_t_jax, suffix_out_jax = jax_one_step(jax.random.PRNGKey(0), jax_obs, noise_jax[0:1], time_jax)
    v_t_jax_np = np.asarray(v_t_jax.astype(jnp.float32))[0]  # (horizon, adim)
    suffix_out_jax_np = np.asarray(suffix_out_jax.astype(jnp.float32))[0]
    print(f"JAX v_t   shape={v_t_jax_np.shape}  norm={np.linalg.norm(v_t_jax_np):.3f}  range=[{v_t_jax_np.min():.3f}, {v_t_jax_np.max():.3f}]")
    print(f"JAX hidden shape={suffix_out_jax_np.shape}  norm={np.linalg.norm(suffix_out_jax_np):.3f}")

    print("\n=== PT: single-step forward (no_kvcache path) ===")
    # Need to build PT-style observation. Reuse pt_obs construction.
    pt_obs_dict = {}
    for k, v in obs_pre.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)[None].to(device)
            pt_obs_dict[k] = t
        elif isinstance(v, dict):
            pt_obs_dict[k] = {kk: torch.from_numpy(vv)[None].to(device) for kk, vv in v.items()}
        else:
            pt_obs_dict[k] = v
    pt_observation = _model.Observation.from_dict(pt_obs_dict)

    pt_model.eval()
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, state = pt_model._preprocess_observation(pt_observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = pt_model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_len = prefix_pad_masks.shape[1]

        # Single step at FIXED_TIME with noise
        x_t = noise_pt
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = pt_model.embed_suffix(state, x_t, time_pt)

        # Cast to bf16 if needed
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        if pt_model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        suffix_to_prefix_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        prefix_to_suffix_masks = torch.zeros(batch_size, prefix_len, suffix_len, dtype=torch.bool, device=prefix_pad_masks.device)
        top_row = torch.cat([prefix_att_2d_masks, prefix_to_suffix_masks], dim=2)
        bottom_row = torch.cat([suffix_to_prefix_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks = torch.cat([top_row, bottom_row], dim=1)
        full_att_2d_masks_4d = pt_model._prepare_attention_masks_4d(full_att_2d_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

        outputs_embeds, _ = pt_model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1][:, -horizon:]
        suffix_out_pt_np = suffix_out.detach().float().cpu().numpy()[0]
        suffix_out_for_proj = suffix_out.to(dtype=pt_model.action_out_proj.weight.dtype)
        v_t_pt = pt_model.action_out_proj(suffix_out_for_proj)
        v_t_pt_np = v_t_pt.detach().float().cpu().numpy()[0]

    print(f"PT v_t   shape={v_t_pt_np.shape}  norm={np.linalg.norm(v_t_pt_np):.3f}  range=[{v_t_pt_np.min():.3f}, {v_t_pt_np.max():.3f}]")
    print(f"PT hidden shape={suffix_out_pt_np.shape}  norm={np.linalg.norm(suffix_out_pt_np):.3f}")

    # Compare
    def cos_sim(a, b):
        a = a.flatten(); b = b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    print("\n================================================================================")
    print("COMPARISON (JAX vs PT at fixed t=0.5, fixed noise, synthetic obs)")
    print("================================================================================")
    print(f"hidden_state (suffix, action_horizon tokens):")
    print(f"  cos      = {cos_sim(suffix_out_jax_np, suffix_out_pt_np):+.6f}")
    print(f"  |jax|    = {np.linalg.norm(suffix_out_jax_np):.4f}")
    print(f"  |pt|     = {np.linalg.norm(suffix_out_pt_np):.4f}")
    print(f"  max|diff|= {np.max(np.abs(suffix_out_jax_np - suffix_out_pt_np)):.4f}")
    print(f"v_t (action_out_proj output, what gets integrated):")
    print(f"  cos      = {cos_sim(v_t_jax_np, v_t_pt_np):+.6f}")
    print(f"  |jax|    = {np.linalg.norm(v_t_jax_np):.4f}")
    print(f"  |pt|     = {np.linalg.norm(v_t_pt_np):.4f}")
    print(f"  ratio    = {np.linalg.norm(v_t_pt_np) / (np.linalg.norm(v_t_jax_np) + 1e-10):.3f}")
    print(f"  max|diff|= {np.max(np.abs(v_t_jax_np - v_t_pt_np)):.4f}")
    # First 5 dims
    print(f"\nv_t[0, 0:8]   JAX: {v_t_jax_np[0, :8]}")
    print(f"v_t[0, 0:8]    PT: {v_t_pt_np[0, :8]}")


if __name__ == "__main__":
    main()
