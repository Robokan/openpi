ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libgl1 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      yasm \
      pkg-config \
      git \
      libgnutls28-dev \
      libvpx-dev \
      libopus-dev \
      libvorbis-dev \
      libmp3lame-dev \
      libfreetype-dev \
      libass-dev \
      libaom-dev \
      libdav1d-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

COPY openpi_on_thor/pyproject.toml .

# Set to get precompiled jetson wheels
RUN export PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130 && \
    export PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io && \
    pip install -e .[thor]

# chex (+ toolz) is needed for JAX→PyTorch conversion (transitive dep of flax/jax).
# Install with --no-deps to avoid upgrading jax/numpy from the Jetson builds.
RUN pip install chex --no-deps
RUN pip install toolz --no-deps

# onnxslim: needed for NVFP4 2DQ format conversion during ONNX export.
RUN pip install onnxslim

# lerobot: OpenPi expects a specific commit with the lerobot.common module structure.
RUN pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" --no-deps

# Copy OpenPI source code
COPY . /workspace/

# Apply transformers patches for PI0 model support
RUN cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* \
    /usr/local/lib/python3.12/dist-packages/transformers/ || true

# Apply ONNX/TRT compatibility fixes to modeling_gemma.py (per Jetson AI Lab tutorial Step 6.4).
# Required for NVFP4 quantization: replaces `reshape(*input_shape, -1)` with explicit dimension
# so TensorRT FP4 block quantization can resolve the blocked axis at build time.
RUN python3 /workspace/openpi_on_thor/patches/apply_gemma_fixes.py

# Set PYTHONPATH for OpenPI modules
ENV PYTHONPATH=/workspace/src:/workspace:/workspace/packages/openpi-client/src:$PYTHONPATH

