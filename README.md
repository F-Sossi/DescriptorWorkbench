# DescriptorWorkbench

A comprehensive computer vision research framework for comparing image descriptors with advanced keypoint generation and processing techniques.

## Overview

DescriptorWorkbench is a production-ready research platform for evaluating image descriptors (SIFT, RGBSIFT, HoNC, VGG, DNN-based descriptors) with sophisticated processing techniques including domain-size pooling, stacking, and **revolutionary spatial intersection architecture for cross-detector evaluation**.

### Key Features

- **Spatial Intersection System**: Revolutionary architecture enabling any detector/descriptor combination
- **Cross-Detector Evaluation**: SIFT, Harris, ORB, and KeyNet detectors with mutual spatial correspondence
- **Pure Intersection Sets**: Native detector parameters preserved at spatially matched locations
- **Controlled Comparisons**: Same spatial sampling across descriptor types for fair evaluation
- **Multiple Descriptor Types**: Traditional (SIFT, SURF, ORB) and CNN-based (HardNet, SOSNet) descriptors
- **Database-Driven Workflow**: SQLite-based experiment tracking with full reproducibility
- **Docker Integration**: User-safe containerized development environment
- **YAML Configuration**: Schema-validated experiment configuration system

## Quick Start

### Prerequisites

- **OpenCV 4.12.0+** (with contrib modules for SIFT, VGG, xfeatures2d)
- **CMake 3.15+**
- **C++17 compiler**
- **SQLite3**
- **Boost 1.70+**
- **yaml-cpp**

### Build Instructions

```bash
# Clone and build
git clone <repository-url>
cd DescriptorWorkbench
mkdir build && cd build

# Configure with system packages (recommended). Database integration is always enabled.
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF

# Build (all tests should pass: 20/20)
make -j$(nproc)

# Verify installation
ctest --output-on-failure
# Expected: 100% tests passed, 0 tests failed out of 20
```

### Native Installation (Platform-Specific)

#### Manjaro/Arch Linux
```bash
# Install system dependencies
sudo pacman -S base-devel cmake git python python-pip
sudo pacman -S opencv boost tbb intel-tbb sqlite yaml-cpp

# Install OpenCV contrib (for SIFT support)
yay -S opencv-contrib  # or paru -S opencv-contrib

# Clone and build
git clone <repository-url>
cd DescriptorWorkbench
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
# Configure (database support builds automatically)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
make -j$(nproc)
```

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install build-essential cmake git python3 python3-pip
sudo apt install libopencv-dev libopencv-contrib-dev
sudo apt install libboost-all-dev libtbb-dev libsqlite3-dev libyaml-cpp-dev

# Clone and build
git clone <repository-url>
cd DescriptorWorkbench
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
# Configure (database support builds automatically)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
make -j$(nproc)
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake opencv boost tbb python3 sqlite yaml-cpp

# Clone and build
git clone <repository-url>
cd DescriptorWorkbench
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
# Configure (database support builds automatically)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
make -j$(nproc)
```

#### Windows (Advanced Users)

**Option 1: Docker (Recommended for Windows)**
```powershell
# Install Docker Desktop, then follow Docker instructions above
```

**Option 2: Native Windows with vcpkg**
```powershell
# Install Visual Studio 2022 with C++ workload
# Install vcpkg package manager
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Install dependencies via vcpkg
.\vcpkg install opencv[contrib]:x64-windows
.\vcpkg install boost:x64-windows
.\vcpkg install tbb:x64-windows
.\vcpkg install yaml-cpp:x64-windows

# Clone and build project
git clone <repository-url>
cd DescriptorWorkbench
python setup.py  # Download HPatches dataset
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

**Option 3: WSL2 (Linux Subsystem)**
```bash
# Install WSL2 and Ubuntu from Microsoft Store
# Follow Ubuntu installation instructions inside WSL2
sudo apt update && sudo apt install build-essential cmake git python3 python3-pip
sudo apt install libopencv-dev libopencv-contrib-dev libboost-all-dev libtbb-dev libsqlite3-dev
# Continue with Ubuntu instructions above
```

### Docker Development (Recommended)

#### Development Environment Features
- **Reproducible builds** across all systems
- **Pre-configured OpenCV** with SIFT support
- **All dependencies included**
- **X11 forwarding** for GUI debugging
- **Volume mounting** for live code editing
- **User-safe permissions** (no root-owned files)

```bash
# Development environment (user-safe, no permission conflicts)
export USER_ID=$(id -u) && export GROUP_ID=$(id -g)
docker-compose -f docker-compose.dev.yml build --no-cache
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml exec descriptor-dev bash

# Inside container
cd /workspace/build
cmake .. -DUSE_SYSTEM_PACKAGES=ON && make -j$(nproc)
```

#### Platform-Specific Docker Setup

**Linux (Manjaro/Arch)**:
```bash
sudo pacman -S docker docker-compose
sudo usermod -aG docker $USER
sudo systemctl start docker && sudo systemctl enable docker
# Log out and back in for group changes
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
# Log out and back in
```

**macOS**:
```bash
brew install docker docker-compose
# OR install Docker Desktop from https://www.docker.com/products/docker-desktop/
```

**Windows**:
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Enable WSL2 integration if using WSL2
# Ensure virtualization in BIOS if needed
```

## Keypoint Generation & Management

### Spatial Intersection System - NEW

The project features a revolutionary **spatial intersection architecture** that enables any detector/descriptor combination through spatially matched keypoint pairs:

```bash
cd build

# 1. Generate max detector sets
./keypoint_manager generate-detector ../data sift sift_independent_full
./keypoint_manager generate-detector ../data orb orb_independent_full

# 2. Create spatial intersection pairs (within 10px tolerance)
./keypoint_manager build-intersection --source-a orb_independent_full --source-b sift_independent_full \
  --out-a orb_sift_pairs --out-b sift_orb_pairs --tolerance 10

# 3. Verify intersection results
./keypoint_manager list-sets
# Shows: orb_sift_pairs (11,093 keypoints), sift_orb_pairs (11,093 keypoints)
```

### Cross-Detector Evaluation Capabilities

**Spatial Intersection Results**:
- **KeyNet-SIFT**: 111,030 matched pairs (33% detector overlap)
- **ORB-SIFT**: 11,093 matched pairs (31% detector overlap)
- **Native Parameters**: Each intersection set retains its detector's optimal keypoint properties
- **Controlled Evaluation**: Same spatial locations across descriptor types for fair comparison

### Legacy Commands (Still Supported)

```bash
# Generate fresh keypoints from images (SIFT, legacy method)
./keypoint_manager generate ../data

# Import/export for reproducibility
./keypoint_manager import-csv ../reference_keypoints
./keypoint_manager export-csv ./exported_keypoints

# Database inspection
./keypoint_manager list-scenes
./keypoint_manager count i_dome 1.ppm
```

## Running Experiments

### YAML-Based Experiment Runner

```bash
cd build

# Run baseline experiments with spatial intersection sets
./experiment_runner ../config/experiments/sift_systematic_analysis.yaml
./experiment_runner ../config/experiments/orb_baseline.yaml

# Cross-detector evaluation experiments
./experiment_runner ../config/experiments/sift_vs_hardnet_keynet.yaml
./experiment_runner ../config/experiments/libtorch_hardnet_baseline.yaml
```

### Dataset Setup

The framework expects HPatches dataset in `data/` directory:

```bash
# Download HPatches dataset automatically
python3 setup.py
```

## Architecture Overview

### Core Components

- **`src/interfaces/`**: Interface definitions (`IKeypointGenerator`, modular design)
- **`src/core/keypoints/`**: Keypoint detector implementations (SIFT, Harris, ORB)
- **`src/core/keypoints/detectors/`**: Concrete detector classes with non-overlapping support
- **`src/core/descriptor/`**: Modern descriptor factory and wrappers
- **`src/core/database/`**: SQLite-based experiment tracking
- **`cli/`**: Command-line tools (experiment_runner, keypoint_manager, analysis_runner)
- **`config/experiments/`**: YAML experiment configurations

### Keypoint Interface Architecture

```cpp
// Core interface supporting all detectors
class IKeypointGenerator {
public:
    virtual std::vector<cv::KeyPoint> detect(const cv::Mat& image, const KeypointParams& params) = 0;
    virtual std::vector<cv::KeyPoint> detectNonOverlapping(const cv::Mat& image, float min_distance, const KeypointParams& params) = 0;
    virtual std::string name() const = 0;
    virtual KeypointGenerator type() const = 0;
};

// Factory pattern for unified creation
auto detector = KeypointGeneratorFactory::create(KeypointGenerator::SIFT, true, 32.0f);
auto keypoints = detector->detectNonOverlapping(image, 32.0f, params);
```

## Database Integration

### Experiment Tracking

All experiments are automatically tracked in SQLite database:

```bash
# View database contents
cd build
sqlite3 experiments.db
.schema                         # View table structure
SELECT * FROM experiments;      # View experiment configs
SELECT * FROM results;          # View experiment results
SELECT * FROM keypoint_sets;    # View keypoint metadata
```

### Database Schema

- **`experiments`**: Experiment configurations and timing
- **`results`**: Precision/recall metrics by scene and descriptor
- **`keypoint_sets`**: Keypoint metadata with overlap filtering parameters

## Supported Descriptors

### Working Descriptor Types

- **SIFT**: Traditional SIFT (OpenCV implementation)
- **RGBSIFT**: Color-aware SIFT variant
- **HoNC**: Histogram of Normalized Colors  
- **VSIFT**: Variant SIFT implementation
- **DSPSIFT**: Domain-Size Pooling SIFT
- **VGG**: VGG descriptors (requires OpenCV contrib)
- **DNN Patch**: Neural network-based descriptors via ONNX

### Pooling Strategies

- **None**: No pooling (baseline)
- **Domain-Size Pooling (DSP)**: Advanced spatial pooling technique
- **Stacking**: Multiple descriptor combination

## Testing

### Test Suite Status

```bash
cd build
ctest --output-on-failure
# Expected: 100% tests passed, 0 tests failed out of 20

# Run specific test groups
make run_database_tests    # Database integration tests
make run_interface_tests   # Interface and keypoint tests
make run_config_tests      # YAML configuration tests
```

### Test Coverage

- **20 comprehensive test suites** covering all major components
- **Interface tests**: Keypoint generator functionality and non-overlapping algorithms
- **Database tests**: SQLite integration and schema validation
- **Configuration tests**: YAML schema validation and defaults
- **Integration tests**: End-to-end workflow validation

## Development

### Project Status (September 2025)

**PRODUCTION-READY STATE** - All systems operational:

- **Build System**: All tests passing (20/20)
- **CLI Tools**: experiment_runner, keypoint_manager, analysis_runner
- **Database Integration**: Full SQLite-based experiment tracking
- **Keypoint Interface Upgrade**: Complete with performance validation
- **Docker Support**: User-safe containerized development environment

### Recent Completion: Keypoint Interface Upgrade (2025-09-12)

- **Interface Implementation**: Complete `IKeypointGenerator` with multiple detector support
- **Non-Overlapping Algorithm**: Greedy spatial filtering for CNN optimization
- **Performance Validation**: +6.29% P@1 and +1.33% MAP improvement achieved
- **Database Schema Migration**: Extended with overlap tracking parameters
- **CLI Integration**: New commands integrated with existing workflow
- **Unit Testing**: 14 comprehensive tests added to build system

### Contributing

1. **Build verification**: Ensure all 20 tests pass
2. **Code style**: Follow existing patterns and conventions
3. **Documentation**: Update relevant sections for new features
4. **Testing**: Add comprehensive tests for new functionality

## YAML Configuration (Schema v1)

### Configuration Format

The framework uses strict Schema v1 for all experiment configurations. Create YAML files in `config/experiments/`:

```yaml
experiment:
  name: "my_experiment"
  description: "Custom SIFT experiment with pooling"
  version: "1.0"
  author: "researcher"

dataset:
  type: "hpatches"
  path: "../data/"
  scenes: ["i_dome", "v_wall"]  # Optional: subset of scenes

keypoints:
  generator: "sift"           # sift|harris|orb|locked_in
  max_features: 1500
  source: "homography_projection"  # or "independent_detection"
  
descriptors:
  - name: "sift_baseline"
    type: "sift"              # sift|rgbsift|vsift|honc|dspsift|vgg
    pooling: "none"           # none|domain_size_pooling|stacking
    use_color: false
    normalize_after_pooling: true
    
  - name: "sift_with_dsp"
    type: "sift"
    pooling: "domain_size_pooling"
    scales: [0.85, 1.0, 1.3]           # Domain size multipliers
    scale_weighting: "gaussian"        # gaussian|triangular|uniform
    scale_weight_sigma: 0.15          # Weighting parameter
    normalize_after_pooling: true

evaluation:
  matching:
    method: "brute_force"     # Matching algorithm
    norm: "l2"                # l2|l1 distance norm
    cross_check: true         # Enable cross-checking
    threshold: 0.8            # Matching threshold [0,1]
  validation:
    method: "homography"      # Validation method
    threshold: 0.05           # Pixel tolerance
    min_matches: 10           # Minimum matches required

database:
  connection: "sqlite:///experiments.db"
  save_keypoints: false
  save_descriptors: false
  save_matches: false
  save_visualizations: true

```

Database tracking is always enabled; toggle individual persistence knobs with the `save_*` flags shown above.

### Template Configurations

Use templates from `config/defaults/` as starting points:
- `minimal.yaml`: Basic SIFT baseline
- `sift_baseline.yaml`: Complete SIFT configuration
- `dsp_gaussian.yaml`: SIFT with Domain Size Pooling
- `stacking_sift_rgbsift.yaml`: Descriptor stacking example

### Configuration Guidelines

**Keypoint Sources**:
- `homography_projection`: Locked-in evaluation, isolates descriptor quality (recommended for controlled studies)
- `independent_detection`: Realistic pipeline, includes detector variance (for end-to-end performance)

**Pooling Strategies**:
- `none`: Fastest, good baseline
- `domain_size_pooling`: Set 3-5 scales around 1.0, prefer gaussian weighting with sigma ~0.15
- `stacking`: Combine complementary descriptors, set `secondary_descriptor` and `stacking_weight`

**Matching Settings**:
- Use `brute_force` with `l2` norm for SIFT-family descriptors
- Start with `threshold: 0.8`, adjust per dataset
- Enable `cross_check: true` for better precision

## Common Issues & Troubleshooting

### Build Issues

#### OpenCV SIFT Not Available
```bash
# Verify SIFT support
python3 -c "import cv2; print('SIFT available:', hasattr(cv2, 'SIFT_create'))"

# If false, install OpenCV contrib:
# Manjaro: yay -S opencv-contrib
# Ubuntu: sudo apt install libopencv-contrib-dev
# macOS: brew install opencv (includes contrib)
```

#### CMake Cache Issues
```bash
# Clean build directory completely
rm -rf build/ cmake-build-*
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON && make -j$(nproc)
```

#### VGGWrapper Linker Errors
```bash
# Ensure OpenCV contrib is installed and detected
cmake .. -DUSE_SYSTEM_PACKAGES=ON
# Look for "OpenCV xfeatures2d detected" in output
```

#### Missing Dependencies
```bash
# Use system packages (recommended)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF

# Install missing packages per platform instructions above
```

### Docker Issues

#### Container Won't Start
```bash
# Check Docker service status
sudo systemctl status docker
sudo systemctl start docker

# Rebuild image from scratch
docker-compose -f docker-compose.dev.yml build --no-cache
docker-compose -f docker-compose.dev.yml up -d
```

#### Permission Issues
```bash
# Ensure user in docker group
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect

# Fix file permissions if needed
sudo chown -R $USER:$USER .
```

#### X11 Forwarding Issues (Linux)
```bash
# Allow X11 connections
xhost +local:docker

# Test X11 in container
docker-compose -f docker-compose.dev.yml run descriptor-dev python3 -c "
import cv2; import numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
cv2.imshow('Test', img); cv2.waitKey(1000)"
```

### Dataset Issues

#### Dataset Not Found
```bash
# Verify dataset structure
ls data/
# Should show: i_ajuntament, v_wall, etc.

# Re-download if corrupted
rm -rf data/
python3 setup.py
```

#### Permission Issues
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

### Runtime Issues

#### Database Connectivity
```bash
# Verify database file exists
ls build/experiments.db

# Check database permissions
chmod 644 build/experiments.db

# Reconfigure to regenerate database schema (integration is always enabled)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
```

#### ONNX Model Loading
```bash
# Check model files exist
ls models/*.onnx

# Verify ONNX runtime available (if using DNN descriptors)
python3 -c "import onnxruntime; print('ONNX Runtime available')"
```

#### Keypoint Generation Fails
```bash
# Generate keypoints with debug output
./keypoint_manager generate ../data --verbose

# Check image file permissions
find data/ -name "*.ppm" -ls | head -5
```

### Windows-Specific Issues

#### Docker Desktop Setup
```powershell
# Enable WSL2 backend
# Settings → General → Use WSL2 based engine

# Ensure virtualization enabled in BIOS
# Enable Hyper-V if needed
```

#### Visual Studio Build Issues
```powershell
# Ensure correct components installed:
# - MSVC v143 compiler toolset  
# - Windows 10/11 SDK
# - CMake tools for C++

# Set vcpkg toolchain if using vcpkg
$env:CMAKE_TOOLCHAIN_FILE = "C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"
```

#### Path Length Issues
```powershell
# Enable long path support (Windows 10+, run as Administrator)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Performance Issues
- **Slow descriptor extraction**: Use fewer keypoints (`max_features: 500`) for initial testing
- **Memory usage**: Monitor with large datasets, consider batch processing
- **Build performance**: Use `-j$(nproc)` for parallel compilation

## License

[Add license information here]

## Citation

[Add citation information for academic use]

## Contact

[Add contact information or contribution guidelines]
