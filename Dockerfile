# Multi-stage build for descriptor research
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    pkg-config \
    wget \
    curl \
    # OpenCV dependencies
    libopencv-dev \
    libopencv-contrib-dev \
    opencv-data \
    python3-opencv \
    # Boost
    libboost-all-dev \
    # Threading
    libtbb-dev \
    # X11 for potential GUI operations
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libxtst-dev \
    # ==================== ADD SQLITE3 HERE ====================
    # SQLite3 development package
    libsqlite3-dev \
    sqlite3 \
    # YAML-cpp for experiment configuration
    libyaml-cpp-dev \
    # ==========================================================
    # Additional utilities
    htop \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Create user matching host user to avoid permission issues
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID dev && \
    useradd -m -u $USER_ID -g dev dev

# Install Python packages for research
# Fix: Quote the numpy version constraint
RUN pip3 install \
    "numpy<2" \
    matplotlib \
    pandas \
    seaborn \
    scikit-learn \
    jupyter \
    pybind11 \
    torch \
    torchvision \
    conan \
    --no-cache-dir

# Development stage - includes additional tools
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    && rm -rf /var/lib/apt/lists/*

# Setup Conan (must be done as root for system-wide config)
RUN conan profile detect --force

# Switch to development user for all subsequent operations
USER dev

# Create working directory
WORKDIR /workspace

# Set environment for OpenCV
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig
ENV DISPLAY=:0

# Default command for development
CMD ["bash"]

# Production stage - minimal runtime with CLI tools
FROM base AS production

# Switch to non-root user for security
USER dev

WORKDIR /workspace

# Copy built binaries (assumes they exist in build/)
COPY build/experiment_runner ./
COPY build/keypoint_manager ./
COPY build/analysis_runner ./

# Copy configuration files
COPY config/ ./config/

# Default command uses modern CLI experiment runner
CMD ["./experiment_runner", "config/experiments/sift_baseline.yaml"]