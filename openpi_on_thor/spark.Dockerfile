# Dockerfile for DGX Spark (GB10) - adapted from thor.Dockerfile
# Uses standard PyTorch NGC container (already has TensorRT + ModelOpt)

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libgtk-3-0 \
      libtbb12 \
      libgl1 \
      libopenblas-dev \
      git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Downgrade numpy to 1.x for compatibility with ModelOpt and other NGC packages
RUN pip install --no-cache-dir 'numpy<2' --force-reinstall

# Core ML dependencies
RUN pip install --no-cache-dir \
    tyro \
    safetensors \
    einops \
    sentencepiece \
    transformers \
    diffusers \
    accelerate \
    websockets \
    nvtx \
    onnxslim \
    pillow

# JAX ecosystem (for config loading) - install with --no-deps to avoid conflicts
RUN pip install --no-cache-dir jax --no-deps
RUN pip install --no-cache-dir jaxlib --no-deps || true
RUN pip install --no-cache-dir flax --no-deps
RUN pip install --no-cache-dir chex --no-deps
RUN pip install --no-cache-dir toolz --no-deps
RUN pip install --no-cache-dir orbax-checkpoint --no-deps || true

# msgpack for websocket protocol
RUN pip install --no-cache-dir msgpack msgpack-numpy

# Set up environment
ENV PYTHONPATH=/workspace/packages/openpi-client/src:/workspace/src:/workspace:$PYTHONPATH

# Copy OpenPI source
COPY . /workspace/

# Apply transformers patches at build time
RUN cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* \
    /usr/local/lib/python3.12/dist-packages/transformers/ || true
