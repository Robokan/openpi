from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    # Reserved keys in the observation dict that carry Real-Time Chunking
    # (RTC) inputs. These bypass the input-transform pipeline (so the prev
    # chunk isn't normalized again, etc.) and are routed to the model's
    # ``realtime_sample_actions`` path.
    #
    # Client API:
    #   obs["_rtc_prev_chunk"]:               np.ndarray, shape (H, action_dim_model)
    #   obs["_rtc_inference_delay"]:          int          (d, controller timesteps consumed)
    #   obs["_rtc_prefix_attention_horizon"]: int, optional (default = H, full overlap)
    #   obs["_rtc_schedule"]:                 str, optional (default "exp")
    #   obs["_rtc_max_guidance_weight"]:      float, optional (default 5.0)
    #
    # The server returns the next chunk plus a "_rtc_chunk_model_space" key
    # that the client must echo back on the following inference call. See
    # ``openpi_client.AsyncActionChunkBroker`` for the canonical client.
    _RTC_KEYS = (
        "_rtc_prev_chunk",
        "_rtc_inference_delay",
        "_rtc_prefix_attention_horizon",
        "_rtc_schedule",
        "_rtc_max_guidance_weight",
    )

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Extract RTC inputs BEFORE applying input transforms. These are not
        # normal observation fields and must not be passed through Normalize.
        rtc_kwargs: dict[str, Any] = {}
        if isinstance(obs, dict):
            for k in self._RTC_KEYS:
                if k in obs:
                    rtc_kwargs[k] = obs.pop(k)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        # Dispatch: if the client supplied an RTC prev chunk and we're on the
        # PyTorch path, use realtime_sample_actions. Otherwise fall through to
        # the existing sample_actions path.
        use_rtc = (
            self._is_pytorch_model
            and rtc_kwargs.get("_rtc_prev_chunk") is not None
            and hasattr(self._model, "realtime_sample_actions")
        )
        if use_rtc:
            prev = np.asarray(rtc_kwargs["_rtc_prev_chunk"], dtype=np.float32)
            if prev.ndim == 2:
                prev = prev[None, ...]
            prev_t = torch.from_numpy(prev).to(self._pytorch_device)
            d = int(rtc_kwargs.get("_rtc_inference_delay", 0))
            pah = int(rtc_kwargs.get("_rtc_prefix_attention_horizon", self._model.config.action_horizon))
            schedule = str(rtc_kwargs.get("_rtc_schedule", "exp"))
            beta = float(rtc_kwargs.get("_rtc_max_guidance_weight", 5.0))
            actions = self._model.realtime_sample_actions(
                self._pytorch_device,
                observation,
                prev_action_chunk=prev_t,
                inference_delay=d,
                prefix_attention_horizon=pah,
                prefix_attention_schedule=schedule,
                max_guidance_weight=beta,
                noise=sample_kwargs.get("noise"),
                num_steps=sample_kwargs.get("num_steps", 10),
            )
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time

        # Capture the raw model-space chunk BEFORE output transforms strip /
        # remap / unnormalize. The client needs this exact tensor as the prev
        # chunk on the next call.
        if self._is_pytorch_model:
            raw_actions_model_space = (
                np.asarray(actions[0, ...].detach().cpu()) if isinstance(actions, torch.Tensor) else np.asarray(actions[0, ...])
            )
        else:
            raw_actions_model_space = np.asarray(actions[0, ...])

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        # Always return the model-space chunk so the client can use RTC on
        # the next call. (Cheap: a few KB.)
        outputs["_rtc_chunk_model_space"] = raw_actions_model_space
        if use_rtc:
            outputs["_rtc_used"] = True
            outputs["_rtc_inference_delay"] = d
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
