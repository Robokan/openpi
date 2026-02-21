# Dockerfile for serving a PI policy using NGC-optimized JAX.
# Based on NVIDIA NGC JAX container with optimized CUDA/cuDNN/XLA.
#
# NGC JAX 25.04 includes:
#   - JAX 0.5.3 (matches openpi requirements)
#   - Flax 0.10.5
#   - Python 3.12
#   - CUDA 12.9.0
#   - cuDNN 9.9.0
#   - Optimized XLA compiler
#
# Build the container:
#   docker build . -t openpi_server_ngc -f scripts/docker/serve_policy_ngc.Dockerfile
#
# Run the container:
#   docker run --rm -it --network=host -v .:/app --gpus=all openpi_server_ngc /bin/bash

FROM nvcr.io/nvidia/jax:25.04-py3

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Create a venv that inherits system site-packages (preserves NGC's JAX)
ENV UV_PROJECT_ENVIRONMENT=/opt/openpi-venv
RUN uv venv --python python3 --system-site-packages $UV_PROJECT_ENVIRONMENT

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src

# First, uninstall conflicting system packages AND remove any leftover files
# Critical version conflicts: jaxtyping, transformers, huggingface-hub, lerobot
RUN pip uninstall -y jaxtyping transformers huggingface-hub lerobot 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.12/dist-packages/lerobot* 2>/dev/null || true

# Install dependencies with correct versions
# Note: --system-site-packages means NGC's JAX will be available
RUN --mount=type=cache,target=/root/.cache/uv \
    . $UV_PROJECT_ENVIRONMENT/bin/activate && \
    GIT_LFS_SKIP_SMUDGE=1 pip install \
    augmax dm-tree einops equinox flatbuffers fsspec[gcs] \
    gym-aloha imageio ml_collections "numpy>=1.22.4,<2.0.0" numpydantic \
    opencv-python orbax-checkpoint pillow sentencepiece \
    torch torchvision tqdm-loggable typing-extensions tyro wandb \
    filelock beartype treescope rich polars pytest \
    "jaxtyping==0.2.36" "transformers==4.53.2" "protobuf>=6.31.1" \
    "huggingface-hub>=0.30.0,<1.0" \
    && GIT_LFS_SKIP_SMUDGE=1 pip install --no-cache-dir "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" \
    && pip install -e packages/openpi-client \
    && python -c "import lerobot; print(f'lerobot installed at: {lerobot.__file__}')" \
    && python -c "import datasets; print(f'datasets version: {datasets.__version__}')"

# Patch lerobot to handle newer datasets Column API
# The datasets library returns Column objects instead of lists, breaking torch.stack
RUN LEROBOT_PATH=$(python -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))") && \
    sed -i 's/torch.stack(self.hf_dataset\["timestamp"\])/torch.tensor(list(self.hf_dataset["timestamp"]))/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" && \
    sed -i 's/torch.stack(self.hf_dataset\["episode_index"\])/torch.tensor(list(self.hf_dataset["episode_index"]))/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" && \
    sed -i 's/torch.stack(self.hf_dataset.select(q_idx)\[key\])/torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in list(self.hf_dataset.select(q_idx)[key])])/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py"

# Set environment
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy transformers_replace files for PyTorch support
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# Verify NGC JAX is accessible
RUN python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

# Default command
CMD ["python", "scripts/serve_policy.py"]
