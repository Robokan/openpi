# Dockerfile for serving LeRobot Diffusion Policy on GB10 (Blackwell)
# Uses NGC PyTorch 26.01 which has full sm_121 support
#
# Build:
#   docker build -f scripts/docker/diffusion_policy.Dockerfile -t diffusion_policy_server .
#
# Run:
#   docker run --rm -it --gpus all --ipc=host -v .:/app -p 8001:8001 diffusion_policy_server \
#       python scripts/serve_diffusion_policy.py --checkpoint /app/outputs/train/diffusion_openarm_v4/checkpoints/last/pretrained_model

FROM nvcr.io/nvidia/pytorch:26.01-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Install LeRobot and dependencies
RUN pip install --upgrade pip && \
    pip install \
    websockets \
    msgpack \
    safetensors \
    "huggingface-hub>=0.30.0" \
    && GIT_LFS_SKIP_SMUDGE=1 pip install --no-cache-dir \
    "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# Copy the serve script (will be mounted, but copy for standalone use)
COPY scripts/serve_diffusion_policy.py /app/scripts/

# Default command
CMD ["python", "scripts/serve_diffusion_policy.py", "--help"]
