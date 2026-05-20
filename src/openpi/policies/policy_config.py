import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def _pytorch_warmup(model, device: str, n_iters: int = 2) -> None:
    """Run dummy sample_actions calls to trigger torch.compile autotune at
    server boot instead of on the first robot WebSocket call.

    Uses the model's `inputs_spec` to fabricate a synthetic observation with
    the correct shapes/dtypes, then matches the dtype expected by the action
    head (handles fp32 vs bf16 / quantized configurations).
    """
    import jax  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    observation_spec, _ = model.config.inputs_spec(batch_size=1)

    def _to_torch(spec):
        # Build a torch tensor matching the JAX ShapeDtypeStruct shape/dtype.
        dt_map = {jnp.float32: torch.float32, jnp.int32: torch.int32, jnp.bool_: torch.bool}
        torch_dtype = dt_map.get(spec.dtype.type, torch.float32)
        if torch_dtype is torch.bool:
            t = torch.ones(spec.shape, dtype=torch.bool, device=device)
        elif torch_dtype.is_floating_point:
            t = torch.zeros(spec.shape, dtype=torch_dtype, device=device)
        else:
            t = torch.zeros(spec.shape, dtype=torch_dtype, device=device)
        return t

    pt_obs_dict = jax.tree.map(_to_torch, observation_spec)
    # action_in_proj.weight.dtype is the dtype the diffusion path expects on entry.
    # If the model has been cast to bf16 / quantized, all float tensors need to match.
    target_dtype = model.action_in_proj.weight.dtype
    pt_obs_dict = jax.tree.map(
        lambda t: t.to(target_dtype) if (isinstance(t, torch.Tensor) and t.is_floating_point()) else t,
        pt_obs_dict,
    )
    observation = _model.Observation.from_dict(pt_obs_dict)

    horizon = model.config.action_horizon
    adim = model.config.action_dim
    _np.random.seed(0)
    noise = torch.from_numpy(_np.random.randn(1, horizon, adim).astype(_np.float32)).to(device).to(target_dtype)

    with torch.no_grad():
        for i in range(n_iters):
            _ = model.sample_actions(device, observation, noise=noise, num_steps=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logging.info(f"  warmup iter {i+1}/{n_iters} complete")


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        # Allow overriding the inference precision via env var. Useful when the checkpoint
        # was converted from a LoRA-finetuned JAX model: the merged-then-cast bf16 weights
        # can lose the LoRA contribution because the per-weight delta is comparable to the
        # bf16 quantization step at the base weight magnitude. Running in float32 preserves
        # the merged LoRA fully.
        pt_precision = os.environ.get("OPENPI_PYTORCH_PRECISION", "bfloat16")
        if pt_precision not in ("bfloat16", "float32"):
            raise ValueError(f"OPENPI_PYTORCH_PRECISION must be 'bfloat16' or 'float32', got {pt_precision!r}")
        logging.info(f"Casting PyTorch model to {pt_precision}")
        model.paligemma_with_expert.to_bfloat16_for_selected_params(pt_precision)

        # Apply optional quantization AFTER the precision cast so torchao's
        # tensor-subclass weights are not stripped by the cast. Composes with
        # runtime LoRA: base Linear -> FP8/NVFP4, LoRA -> bf16 (additive).
        quant_mode = os.environ.get("OPENPI_PT_QUANT", "").strip()
        if quant_mode:
            from openpi.models_pytorch import quant_runtime  # noqa: PLC0415
            n = quant_runtime.install_quantization(model, quant_mode)
            logging.info(f"Quantization installed: mode={quant_mode}, modules={n}")

        # Eager warmup: drive at least one full sample_actions through the
        # model so torch.compile's max-autotune cost is paid at server boot,
        # not on the robot's first WebSocket call. Without this the robot
        # appears frozen for 60-180s on the first observation (longer with
        # FP8 quant, where there are more kernel candidates to benchmark).
        # Opt out with OPENPI_PT_SKIP_WARMUP=1.
        if os.environ.get("OPENPI_PT_SKIP_WARMUP", "0") != "1":
            import time as _time  # noqa: PLC0415
            import torch as _torch  # noqa: PLC0415
            n_warmup = int(os.environ.get("OPENPI_PT_WARMUP_ITERS", "2"))
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            logging.info(f"Warming up PyTorch model on {device} with {n_warmup} sample_actions iters...")
            t0 = _time.time()
            try:
                _pytorch_warmup(model, device=device, n_iters=n_warmup)
                logging.info(f"PyTorch warmup completed in {_time.time()-t0:.1f}s. Server now ready for first robot call.")
            except Exception as e:
                logging.warning(f"PyTorch warmup failed (server will still start but first call will autotune): {e}")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
