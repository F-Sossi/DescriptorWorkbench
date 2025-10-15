# Multi-stage build for descriptor research
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip pkg-config wget curl \
    libboost-all-dev libtbb-dev libeigen3-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenexr-dev libwebp-dev \
    libgtk-3-dev libcanberra-gtk3-module libdc1394-dev libv4l-dev \
    libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev \
    libx11-dev libxext-dev libxrender-dev libxtst-dev \
    libsqlite3-dev sqlite3 libyaml-cpp-dev \
    libgtest-dev ninja-build htop vim nano \
    && rm -rf /var/lib/apt/lists/*

# Build GoogleTest (Ubuntu ships sources only)
RUN cmake -S /usr/src/googletest -B /usr/src/googletest/build && \
    cmake --build /usr/src/googletest/build && \
    cmake --install /usr/src/googletest/build

# Build OpenCV with contrib (includes xfeatures2d / nonfree)
ENV OPENCV_VERSION=4.10.0
RUN mkdir -p /tmp/opencv-build && cd /tmp/opencv-build && \
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -O opencv.tar.gz && \
    wget -q https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz -O opencv_contrib.tar.gz && \
    tar -xf opencv.tar.gz && tar -xf opencv_contrib.tar.gz && \
    mkdir -p opencv-${OPENCV_VERSION}/build && cd opencv-${OPENCV_VERSION}/build && \
    cmake -G Ninja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv-build/opencv_contrib-${OPENCV_VERSION}/modules \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_opencv_python3=ON \
        -D BUILD_opencv_java=OFF \
        -D WITH_CUDA=OFF \
        -D WITH_OPENCL=OFF \
        -D WITH_IPP=ON \
        .. && \
    cmake --build . --target install && \
    ldconfig && \
    rm -rf /tmp/opencv-build

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
