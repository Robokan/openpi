# PyTorch NGC container for LeRobot Diffusion Policy training on DGX Spark
# Based on NVIDIA's PyTorch container with GB10/Blackwell support

FROM nvcr.io/nvidia/pytorch:25.04-py3

WORKDIR /app

# Install git-lfs and other dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Remove ALL NGC pip constraints that conflict with LeRobot
ENV PIP_CONSTRAINT=""
RUN rm -f /opt/nvidia/pip-constraints.txt /opt/nvidia/entrypoint.d/80-pip-constraints.sh 2>/dev/null || true && \
    pip config unset global.constraint 2>/dev/null || true && \
    rm -f /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null || true

# Match JAX container versions: datasets==4.5.0, numpy==1.26.4
# Keep numpy<2 to be compatible with NGC PyTorch
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
    pip install --no-cache-dir \
    "datasets==4.5.0" \
    "diffusers>=0.28.0" \
    dm-env \
    gymnasium \
    h5py \
    "huggingface_hub>=0.30.0,<1.0" \
    "hydra-core>=1.3.2" \
    imageio \
    jsonlines \
    omegaconf \
    opencv-python \
    av \
    termcolor \
    wandb \
    zarr \
    draccus \
    safetensors \
    accelerate \
    msgpack \
    msgpack-numpy \
    websockets

# Install same LeRobot commit as JAX container (without deps to preserve PyTorch)
RUN GIT_LFS_SKIP_SMUDGE=1 pip install --no-cache-dir --no-deps \
    "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"

# CRITICAL: Force numpy back to 1.26.4 AFTER all installs
# NGC PyTorch was compiled with numpy 1.x and will fail with numpy 2.x
RUN pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

# Patch lerobot to handle newer datasets Column API
# The datasets library returns Column objects instead of lists, breaking torch.stack
RUN LEROBOT_PATH=$(python -c "import lerobot; import os; print(os.path.dirname(lerobot.__file__))") && \
    sed -i 's/torch.stack(self.hf_dataset\["timestamp"\])/torch.tensor(list(self.hf_dataset["timestamp"]))/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" && \
    sed -i 's/torch.stack(self.hf_dataset\["episode_index"\])/torch.tensor(list(self.hf_dataset["episode_index"]))/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" && \
    sed -i 's/torch.stack(self.hf_dataset.select(q_idx)\[key\])/torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in list(self.hf_dataset.select(q_idx)[key])])/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py" && \
    sed -i 's/torch.stack(timestamps).tolist()/torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in list(timestamps)]).tolist()/' \
    "$LEROBOT_PATH/common/datasets/lerobot_dataset.py"

# Mount the HuggingFace cache for datasets
ENV HF_HOME=/root/.cache/huggingface

CMD /bin/bash
