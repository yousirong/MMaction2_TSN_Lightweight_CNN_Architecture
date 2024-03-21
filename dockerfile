# Base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

# Set the working directory
WORKDIR /app/mmaction2

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /app/mmaction2/requirements.txt
RUN pip install -r requirements.txt

# Install OpenMIM
RUN pip install -U openmim

# Install MMEngine
RUN mim install mmengine

# Install MMCV
RUN mim install mmcv

# Install MMDetection
RUN mim install mmdet

# Install other dependencies
RUN pip install timm mmpretrain decord future tensorboard

# Set CUBLAS_WORKSPACE_CONFIG environment variable
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

# Copy your configuration file
COPY ./mmaction2/configs/recognition/tsn/juneyong_backbones/tsn_shufflenet_v2_1x1x8_20e_Dassult.py ./app/mmaction2/tsn_shufflenet_v2_1x1x8_20e_Dassult.py


# Train the modeldocker build -t mmaction2_training .docker build -t mmaction2_training .
CMD ["python", "tools/train.py", "./configs/recognition/tsn/juneyong_backbones/tsn_shufflenet_v2_1x1x8_20e_Dassult.py"]
