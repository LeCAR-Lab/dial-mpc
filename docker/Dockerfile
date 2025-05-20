FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DISPLAY=unix$DISPLAY

# Install necessary packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        sudo \
        software-properties-common \
        python-is-python3 \
        git \
        python3-pip \
        libgl1-mesa-glx \
        libglu1-mesa \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libx11-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
WORKDIR /root/
RUN git clone https://github.com/LeCar-Lab/dial-mpc.git --depth 1

# Install the package
WORKDIR /root/dial-mpc
RUN pip3 install -e .

# Default command
CMD ["/bin/bash"]
