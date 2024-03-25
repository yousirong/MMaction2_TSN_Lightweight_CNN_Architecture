# ARG PYTORCH="2.2.1"
# ARG CUDA="12.4"
# ARG CUDNN="9"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# Base image
FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04 as base

ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && \
    apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and add it to the PATH
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm /tmp/miniconda.sh 
RUN conda update -y conda

# Create Conda environment
ARG PYTHON_VERSION=3.9
ARG CONDA_ENV_NAME=d2l
RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION

# Set environment variables
ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Install OpenMIM
RUN pip install -U openmim

# Install MMEngine
RUN mim install mmengine

# Install MMCV
RUN mim install mmcv

# Install MMDetection
RUN mim install mmdet

# Set environment variable
ENV CUBLAS_WORKSPACE_CONFIG=:16:8

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/yousirong/MMaction2_TSN_Lightweight_CNN_Architecture.git /app
WORKDIR /app
RUN mkdir -p /mmaction2/data
ENV FORCE_CUDA="1"

# Copy MMAction2 source code into the Docker image
COPY . /app

CMD ["bin/bash"]
