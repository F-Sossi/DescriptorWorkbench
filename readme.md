# Descriptor Research Project

A comprehensive computer vision framework for comparing and evaluating image descriptors (SIFT, RGBSIFT, HoNC, etc.) using domain size pooling, stacking, and other advanced processing techniques.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
- [Development Environments](#development-environments)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Option 1: Docker (Recommended - Works Everywhere)
```bash
git clone <your-repository-url>
cd descriptor-compare
python3 setup.py  # Downloads HPatches dataset
docker build -t descriptor-research:dev --target development .
docker-compose -f docker-compose.dev.yml up -d
docker-compose -f docker-compose.dev.yml exec descriptor-dev bash
cd /workspace/build-docker && cmake .. && make -j$(nproc)
./descriptor_compare
```

### Option 2: Native Linux (Manjaro/Arch)
```bash
git clone <your-repository-url>
cd descriptor-compare
sudo pacman -S base-devel cmake opencv boost tbb sqlite
python3 setup.py  # Downloads HPatches dataset
mkdir build && cd build && cmake .. -DBUILD_DATABASE=ON && make -j$(nproc)
./descriptor_compare
```

### Option 3: Windows (Docker - Easiest)
```powershell
git clone <your-repository-url>
cd descriptor-compare
python setup.py  # Downloads HPatches dataset
docker build -t descriptor-research:dev --target development .
docker run -it --rm -v ${PWD}:/workspace descriptor-research:dev bash
# Inside container: cd /workspace/build-docker && cmake .. && make -j$(nproc) && ./descriptor_compare
```

## Installation Methods

### Docker Installation (Recommended for Reproducibility)

#### Prerequisites
```bash
# Install Docker and Docker Compose

# Linux (Manjaro/Arch)
sudo pacman -S docker docker-compose

# Linux (Ubuntu/Debian)
sudo apt install docker.io docker-compose

# macOS
brew install docker docker-compose
# OR install Docker Desktop from https://www.docker.com/products/docker-desktop/

# Windows
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/
# Enable WSL2 integration if using WSL2

# Add user to docker group (Linux only)
sudo usermod -aG docker $USER
sudo systemctl start docker
sudo systemctl enable docker
# Log out and back in for group changes
```

#### Build and Run
```bash
# Clone repository
git clone <your-repository-url>
cd descriptor-compare

# Download dataset (required)
python3 setup.py

# Build Docker environment
docker build -t descriptor-research:dev --target development .

# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Enter development container
docker-compose -f docker-compose.dev.yml exec descriptor-dev bash

# Build project inside container
cd /workspace
mkdir build-docker && cd build-docker
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON -DBUILD_DATABASE=ON
make -j$(nproc)

# Run experiments (database-enabled)
./descriptor_compare
```

#### Docker Environment Features
- ✅ **Reproducible builds** across all systems
- ✅ **Pre-configured OpenCV** with SIFT support
- ✅ **All dependencies included**
- ✅ **X11 forwarding** for GUI debugging
- ✅ **Volume mounting** for live code editing

### Native Installation

#### Manjaro/Arch Linux
```bash
# Install system dependencies
sudo pacman -S base-devel cmake git python python-pip
sudo pacman -S opencv boost tbb intel-tbb sqlite

# Install OpenCV contrib (for SIFT support)
yay -S opencv-contrib  # or paru -S opencv-contrib

# Clone and build
git clone <your-repository-url>
cd descriptor-compare
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)
./descriptor_compare
```

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install build-essential cmake git python3 python3-pip
sudo apt install libopencv-dev libopencv-contrib-dev
sudo apt install libboost-all-dev libtbb-dev libsqlite3-dev

# Clone and build
git clone <your-repository-url>
cd descriptor-compare
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)
./descriptor_compare
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake opencv boost tbb python3 sqlite

# Clone and build
git clone <your-repository-url>
cd descriptor-compare
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)
./descriptor_compare
```

#### Windows

##### Option 1: Docker (Recommended)
```powershell
# Install Docker Desktop from https://www.docker.com/products/docker-desktop/

# Clone repository
git clone <your-repository-url>
cd descriptor-compare

# Download dataset
python setup.py

# Build and run in Docker
docker build -t descriptor-research:dev --target development .
docker run -it --rm -v ${PWD}:/workspace descriptor-research:dev bash

# Inside container:
cd /workspace
mkdir build-docker && cd build-docker
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)
./descriptor_compare
```

##### Option 2: Native Windows (Advanced)
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

# Clone and build project
git clone <your-repository-url>
cd descriptor-compare
python setup.py  # Download HPatches dataset

# Build with Visual Studio
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DUSE_SYSTEM_PACKAGES=ON
cmake --build . --config Release
.\Release\descriptor_compare.exe
```

##### Option 3: WSL2 (Linux Subsystem)
```bash
# Install WSL2 and Ubuntu from Microsoft Store
# Follow Ubuntu installation instructions inside WSL2:

sudo apt update
sudo apt install build-essential cmake git python3 python3-pip
sudo apt install libopencv-dev libopencv-contrib-dev
sudo apt install libboost-all-dev libtbb-dev libsqlite3-dev

git clone <your-repository-url>
cd descriptor-compare
python3 setup.py  # Download HPatches dataset
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON
make -j$(nproc)
./descriptor_compare
```

## Development Environments

### CLion Integration (Recommended for Development)

#### Multi-Environment Setup
1. **Open project in CLion**
2. **Configure Docker Toolchain**:
   - Settings → Build → Toolchains
   - Add Docker toolchain: `Docker Ubuntu 22.04`
   - Image: `descriptor-research:dev`

3. **Create CMake Profiles**:
   - **Native-Debug**: Fast local development
   - **Docker-Debug**: Reproducible container development
   - **Native-Release**: Performance testing
   - **Docker-Release**: Reproducible performance testing

#### CLion CMake Profiles Configuration

**Native-Debug:**
```
Build type: Debug
CMake options: -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
Build directory: cmake-build-debug
```

**Docker-Debug:**
```
Build type: Debug
Toolchain: Docker Ubuntu 22.04
CMake options: -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
Build directory: cmake-build-docker-debug
```

#### Switching Environments
- **Select configuration** from CLion toolbar dropdown
- **Build/Debug/Run** seamlessly in either environment
- **Native**: Full speed, all debugging features
- **Docker**: Reproducible, shareable environment

### Command Line Development

#### Docker Development Workflow
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Develop with live code mounting
docker-compose -f docker-compose.dev.yml exec descriptor-dev bash

# Build and test
cd /workspace/build-docker
make -j$(nproc)
./descriptor_compare

# Stop environment
docker-compose -f docker-compose.dev.yml down
```

#### Native Development Workflow
```bash
# Standard CMake workflow
cd build
make -j$(nproc)
./descriptor_compare

# Or rebuild from scratch
cd .. && rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Usage

### Running Experiments

#### NEW: CLI + YAML Configuration (Recommended)
```bash
cd build
./experiment_runner ../config/experiments/sift_baseline.yaml
./experiment_runner ../config/experiments/rgbsift_comparison.yaml

# The CLI experiment runner will:
# 1. Load YAML experiment configuration
# 2. Extract descriptors using YAML-defined methods
# 3. Evaluate precision using homography validation  
# 4. Save results to results/experiment_name/ directory
```

#### Legacy: Basic Usage
```bash
# Run with hardcoded configuration
./descriptor_compare

# The program will:
# 1. Load HPatches dataset from data/
# 2. Extract descriptors using hardcoded methods
# 3. Evaluate precision using homography validation
# 4. Save results to results/ directory
```

#### Dataset Requirements
The program expects the HPatches dataset in the `data/` directory:
```
data/
├── i_ajuntament/
│   ├── 1.ppm
│   ├── 2.ppm
│   ├── ...
│   ├── H_1_2
│   ├── H_1_3
│   └── ...
└── (other scene folders)
```

**Download dataset automatically:**
```bash
python3 setup.py
```

#### YAML Configuration Format

Create experiment configurations in `config/experiments/`. Example:

```yaml
experiment:
  name: "my_sift_experiment"
  description: "Custom SIFT experiment"

dataset:
  path: "../data/"

descriptors:
  - name: "sift_baseline"
    type: "sift"
    pooling: "none"
  - name: "sift_with_dsp"
    type: "sift"
    pooling: "domain_size_pooling"
    # Optional DSP parameters
    scales: [0.75, 1.0, 1.25]           # domain size multipliers (keypoint-size scaling)
    # Choose one of the following weighting modes:
    # 1) Explicit weights aligned with scales
    # scale_weights: [1.0, 2.0, 1.0]
    # 2) Procedural weighting
    scale_weighting: gaussian            # gaussian|triangular|uniform
    scale_weight_sigma: 0.15            # sigma in log-space (triangular uses as radius proxy)
    # Normalization/RootSIFT stages (row-wise)
    # normalize_before_pooling: false
    # normalize_after_pooling: true
    # rooting options: R_BEFORE_POOLING or R_AFTER_POOLING (via legacy or CLI fields)

evaluation:
  matching:
    method: "brute_force"
    threshold: 0.8
  validation:
    method: "homography"
    threshold: 0.05

output:
  results_path: "results/"
  save_visualizations: true

database:
  enabled: false
```

**Available descriptor types:** `sift`, `rgbsift`, `vsift`, `honc`  
**Available pooling strategies:** `none`, `domain_size_pooling`, `stacking`

DSP-SIFT (Domain Size Pooling):
- Uses keypoint-size scaling over `scales` on the original image (no image resizing).
- Pools per-scale descriptors by weighted average: explicit `scale_weights` or procedural `scale_weighting` + `scale_weight_sigma`.
- Normalizes per row (L2) by default after pooling; RootSIFT supported (L1 then sqrt) before/after.

#### Legacy Configuration Options

**Descriptor Types:**
- `DESCRIPTOR_SIFT`: Standard SIFT
- `DESCRIPTOR_vSIFT`: Vanilla SIFT implementation
- `DESCRIPTOR_RGBSIFT`: RGB color SIFT
- `DESCRIPTOR_HoNC`: Histogram of Normalized Colors

**Processing Options:**
- `NONE`: No additional processing
- `DOMAIN_SIZE_POOLING`: Multi-scale descriptor pooling
- `STACKING`: Descriptor combination

**Modify configuration in:**
- `descriptor_compare/main.cpp` - Main experiment loop
- `descriptor_compare/experiment_config.hpp` - Configuration options

### Expected Output

```
Processing folder: /path/to/data/i_ajuntament
Results folder: /path/to/results/experiment_name/i_ajuntament
Hostname: container_id (if running in Docker)
Processing locked folder: /path/to/data/v_wall
Results folder: /path/to/results/experiment_name/v_wall
...
```

**CLI Results saved to:**
- `results/experiment_name/descriptor_name/scene_name/results.csv`
- Example: `results/sift_baseline/sift/i_dome/results.csv`
- Contains precision measurements for each image pair

**Legacy Results saved to:**
- `results/experiment_name/scene_name/results.csv`
- Contains precision measurements for each image pair

## Database Integration

The project now features a complete database-driven workflow for experiment tracking and keypoint management.

### Database Features

✅ **Real-Time Experiment Tracking**: All experiments automatically tracked in SQLite database  
✅ **Comprehensive Metrics Storage**: Precision, recall, MAP, processing time, metadata  
✅ **Keypoint Management**: Database storage for reproducible locked-in keypoints  
✅ **CLI Tools**: Command-line tools for database and keypoint operations  
✅ **No More CSV Dependencies**: Database-first workflow eliminates CSV file dependencies  

### Database Setup

The database is **automatically enabled** with the default build:

```bash
# Database is enabled by default
cmake .. -DBUILD_DATABASE=ON  # (default)
make -j$(nproc)

# Database file created at: build/experiments.db
```

### Using the Database

#### Viewing Database Contents

```bash
cd build

# SQLite CLI
sqlite3 experiments.db
.schema                         # View table structure
SELECT * FROM experiments;      # View experiment configs  
SELECT * FROM results;          # View experiment results
SELECT * FROM locked_keypoints; # View stored keypoints

# Or use any SQLite GUI tool
```

#### Database Schema

**experiments** table:
- `id`, `descriptor_type`, `dataset_name`, `pooling_strategy`
- `similarity_threshold`, `max_features`, `timestamp`, `parameters`

**results** table:
- `experiment_id`, `mean_average_precision`, `precision_at_1`, `precision_at_5`
- `recall_at_1`, `recall_at_5`, `total_matches`, `total_keypoints`
- `processing_time_ms`, `timestamp`, `metadata`

**locked_keypoints** table:
- `scene_name`, `image_name`, `x`, `y`, `size`, `angle`
- `response`, `octave`, `class_id`, `created_at`

### Keypoint Management System

The project includes a complete CLI tool for managing locked-in keypoints in the database.

#### Keypoint Manager Commands

```bash
cd build

# Generate fresh keypoints from images and store in database
./keypoint_manager generate ../data

# Import keypoints from existing CSV files for reproducibility  
./keypoint_manager import-csv ../reference_keypoints

# Export keypoints from database to CSV for long-term storage
./keypoint_manager export-csv ./exported_keypoints

# List all scenes with keypoint counts
./keypoint_manager list-scenes

# Count keypoints for specific scene/image
./keypoint_manager count i_dome 1.ppm
```

#### Keypoint Workflow Examples

**Generate Fresh Keypoints:**
```bash
# Clear database and generate new keypoints from all images
./keypoint_manager generate ../data

# Output:
# 🔄 Generating fresh locked keypoints from: ../data
# 🗑️  Clearing existing keypoints from database...
# 📁 Processing scene: i_dome
# ✅ Generated 6573 keypoints for i_dome/1.ppm
# ✅ Stored 6573 keypoints for i_dome/2.ppm
# 🎉 Generation complete! Total keypoints generated: 78876
```

**Import from CSV for Reproducibility:**
```bash
# Import pre-computed keypoints from reference files
./keypoint_manager import-csv ../reference_keypoints

# Perfect for:
# - Sharing reproducible experiments
# - Comparing against published results  
# - Starting from known keypoint sets
```

**Export for Long-Term Storage:**
```bash
# Export current database keypoints to CSV files
./keypoint_manager export-csv ./backup_keypoints

# Creates organized CSV structure:
# backup_keypoints/
# ├── i_dome/
# │   ├── 1ppm.csv
# │   ├── 2ppm.csv
# │   └── ...
# └── v_wall/
#     ├── 1ppm.csv
#     └── ...
```

**Query Keypoint Status:**
```bash
# See what's in the database
./keypoint_manager list-scenes

# Output:
# 📋 Available scenes (2):
#   📁 i_dome (6 images, 39438 total keypoints)
#   📁 v_wall (6 images, 41259 total keypoints)

# Get specific counts
./keypoint_manager count i_dome 1.ppm
# 🔢 Keypoints for i_dome/1.ppm: 6573
```

### Database vs CSV Workflow

**OLD Workflow (CSV-dependent):**
```bash
# 1. Generate CSV files manually
# 2. Store keypoints in reference_keypoints/
# 3. Load from CSV during experiments  
# 4. Results saved to CSV files
# 5. Analyze CSV files separately
```

**NEW Workflow (Database-first):**
```bash
# 1. Generate keypoints directly to database
./keypoint_manager generate ../data

# 2. Run experiments with real-time database tracking
./experiment_runner ../config/experiments/sift_baseline.yaml

# 3. Query results directly from database
sqlite3 experiments.db "SELECT mean_average_precision, processing_time_ms FROM results;"

# 4. Export only when needed for sharing
./keypoint_manager export-csv ./published_keypoints
```

### Migration from CSV to Database

If you have existing CSV keypoints:

```bash
# 1. Import your existing CSV keypoints  
./keypoint_manager import-csv ../reference_keypoints

# 2. Verify import
./keypoint_manager list-scenes

# 3. Run experiments (now uses database)
./experiment_runner ../config/experiments/sift_baseline.yaml

# 4. Your CSV files are preserved for compatibility
```

### Troubleshooting Database Issues

#### Database File Not Found
```bash
# Database file is created automatically in build directory
ls build/experiments.db

# If missing, run any experiment to create it
cd build && ./descriptor_compare
```

#### Permission Issues
```bash
# Fix database permissions
chmod 644 build/experiments.db
```

#### View Database Statistics
```bash
# Check experiment history
sqlite3 build/experiments.db "
SELECT 
    e.descriptor_type,
    r.mean_average_precision,
    r.processing_time_ms,
    r.timestamp
FROM experiments e 
JOIN results r ON e.id = r.experiment_id 
ORDER BY r.timestamp DESC 
LIMIT 10;"
```

## Project Structure

```
descriptor-compare/
├── CMakeLists.txt              # Build configuration
├── Dockerfile                  # Docker environment
├── docker-compose.dev.yml     # Development container config
├── setup.py                    # Dataset download script
├── environment.yml             # Conda environment (optional)
├── conanfile.txt              # Conan dependencies (optional)
│
├── cli/                        # NEW: Command-line tools
│   ├── experiment_runner.cpp   # YAML-based experiment runner
│   ├── keypoint_manager.cpp    # Database keypoint management
│   └── analysis_runner.cpp     # Analysis pipeline runner
│
├── config/                     # NEW: YAML experiment configurations
│   └── experiments/
│       ├── sift_baseline.yaml
│       ├── rgbsift_comparison.yaml
│       └── sample_experiment.yaml
│
├── keypoints/                  # Descriptor implementations
│   ├── VanillaSIFT.h/cpp      # Base SIFT implementation
│   ├── DSPSIFT.h/cpp          # Domain Size Pooling SIFT
│   ├── RGBSIFT.h/cpp          # RGB Color SIFT
│   ├── HoNC.h/cpp             # Histogram of Normalized Colors
│   └── HoWH.h/cpp             # Hue-weighted descriptor
│
├── descriptor_compare/         # Main application
│   ├── main.cpp               # Main experiment runner
│   ├── experiment_config.hpp  # Configuration management
│   ├── image_processor.hpp    # Image processing pipeline
│   ├── processor_utils.hpp    # Utility functions
│   └── locked_in_keypoints.hpp # Keypoint management
│
├── database/                  # NEW: Database integration
│   ├── schema.sql             # Database schema definition
│   └── database_manager.cpp   # Legacy database implementation
│
├── src/core/database/         # NEW: Modern database system
│   └── DatabaseManager.cpp    # SQLite integration & keypoint storage
│
├── data/                      # HPatches dataset (created by setup.py)
├── results/                   # Experiment results
├── reference_keypoints/       # Pre-computed keypoints (CSV)
└── build/experiments.db       # SQLite database (auto-created)
```
Current goal state

``` 
 descriptor-research/
  ├── CMakeLists.txt                 # Build system
  ├── setup.py                      # Dataset downloader
  ├── docker-compose.dev.yml        # Development environment
  │
  ├── cli/                          # ✅ ACTIVE: Database-first CLI tools
  │   ├── experiment_runner.cpp     # YAML experiment runner (database-tracked)
  │   ├── keypoint_manager.cpp      # Database keypoint CRUD operations
  │   └── analysis_runner.cpp       # Analysis pipeline
  │
  ├── config/                       # ✅ ACTIVE: YAML experiment configs
  │   └── experiments/
  │       ├── sift_baseline.yaml
  │       ├── rgbsift_comparison.yaml
  │       └── honc_comparison.yaml
  │
  ├── descriptor_compare/           # ✅ ACTIVE: Main application (database-integrated)
  │   ├── main.cpp                  # Database-tracked experiments
  │   ├── image_processor.cpp       # DATABASE KEYPOINT LOADING (not CSV!)
  │   ├── experiment_config.hpp     # Legacy descriptor creation
  │   └── processor_utils.cpp       # Legacy processing pipeline
  │
  ├── keypoints/                    # ✅ ACTIVE: Legacy descriptor implementations
  │   ├── VanillaSIFT.cpp          # Working implementations
  │   ├── RGBSIFT.cpp              # Currently used by main app
  │   ├── HoNC.cpp                 # Working implementations
  │   └── ...
  │
  ├── src/core/                     # ✅ ACTIVE: Modern architecture
  │   ├── database/
  │   │   └── DatabaseManager.cpp   # ✅ ACTIVE: Currently used
  │   ├── descriptor/               # ✅ ACTIVE: New extractor wrappers + factory
  │   │   ├── DescriptorFactory.cpp
  │   │   └── extractors/wrappers/
  │   │       ├── SIFTWrapper.cpp
  │   │       └── RGBSIFTWrapper.cpp
  │   └── pooling/ + matching/ + metrics/  # Core pipeline components
  │
  ├── database/                     # ✅ ACTIVE: Database system
  │   ├── schema.sql               # Current schema definition
  │   └── experiments.db           # Runtime database (auto-created)
  │
  ├── data/                        # ✅ ACTIVE: HPatches dataset
  ├── results/                     # ✅ ACTIVE: Experiment outputs
  ├── reference_keypoints/         # ❌ DEPRECATED: CSV compatibility only
  └── build/experiments.db         # ✅ ACTIVE: Main database file

```

## Schema v1 Quick Start

Use the strict Schema v1 for all configs. Start from the templates in `config/defaults/`.

Minimal runnable example
```yaml
experiment:
  name: "minimal_sift"
  version: "1.0"

dataset:
  type: "hpatches"
  path: "../data/"

keypoints:
  generator: "sift"
  max_features: 1000
  source: "homography_projection"  # or "independent_detection"

descriptors:
  - name: "sift"
    type: "sift"
    pooling: "none"
    normalize_after_pooling: true

evaluation:
  matching: { method: brute_force, norm: l2, cross_check: true, threshold: 0.8 }
  validation: { method: homography, threshold: 0.05, min_matches: 10 }

output: { results_path: "results/", save_visualizations: true }

database:
  enabled: true
  connection: "sqlite:///experiments.db"
```

Key fields (Schema v1)
- experiment: metadata only (name/version/author)
- dataset: `path` to HPatches root; optional `scenes: []` to subset
- keypoints: `generator` (sift|harris|orb|locked_in), `max_features`, `source` (homography_projection|independent_detection)
- descriptors[]: For each descriptor set:
  - `type`: sift|rgbsift|vsift|honc
  - `pooling`: none|domain_size_pooling|stacking
  - Normalization flags: `normalize_before_pooling`, `normalize_after_pooling`
  - DSP: `scales`, `scale_weights` (matches scales len), or `scale_weighting` (uniform|triangular|gaussian) + `scale_weight_sigma > 0`
  - Stacking: `secondary_descriptor`, `stacking_weight` in [0,1]
- evaluation: matching and homography validation settings
- output/database: I/O and tracking

Validation rules (enforced)
- dataset.path required; descriptors non-empty
- Unique `descriptors[].name`
- `stacking_weight` in [0,1]; `scale_weight_sigma` > 0
- Stacking requires `secondary_descriptor`
- DSP `scale_weights` length must equal `scales`; all `scales` > 0
- Matching threshold in [0,1]; keypoint `sigma > 0`, `num_octaves > 0`, `max_features >= 0`

Templates
- `config/defaults/minimal.yaml`: minimal SIFT baseline
- `config/defaults/sift_baseline.yaml`: baseline with explicit thresholds
- `config/defaults/dsp_gaussian.yaml`: SIFT + DSP with gaussian weighting
- `config/defaults/stacking_sift_rgbsift.yaml`: SIFT stacked with RGBSIFT

## Experiment Design Tips

- Keypoint source:
  - `homography_projection`: locked-in evaluation; isolates descriptor quality (recommended for controlled studies).
  - `independent_detection`: realistic pipeline; includes detector variance (use for end-to-end performance).

- Pooling strategies:
  - `none`: fastest; good baseline and ablations.
  - `domain_size_pooling` (DSP): set 3–5 scales around 1.0 (e.g., [0.85, 1.0, 1.3]); prefer `scale_weighting: gaussian` with `scale_weight_sigma ~ 0.15`.
  - `stacking`: combine complementary descriptors (e.g., sift + rgbsift); ensure `secondary_descriptor` and consider `stacking_weight` when scoring.

- Normalization/rooting:
  - Prefer `normalize_after_pooling: true` for most setups.
  - If using RootSIFT-like behavior, apply L1 + sqrt rooting before or after pooling consistently (see pooling_semantics.md) and align with evaluation norm.

- Matching settings:
  - Use `matching: { method: brute_force, norm: l2, cross_check: true }` for SIFT-family descriptors.
  - Start with `threshold: 0.8` for baselines; vary per dataset if needed.

- Color vs grayscale:
  - `use_color: true` for color descriptors (e.g., RGBSIFT); keep false for SIFT/vSIFT.

- Evaluation practice:
  - Always log both legacy precision and true mAP; use homographies where available to compute per-query AP.
  - Report per-scene true mAP to reveal variance across scenes.

Where to look
- See `docs/pooling_semantics.md` for details on weighted pooling and normalization semantics.
- Use `config/defaults/*` as starting points; copy into `config/experiments/` for your runs.

## Supported Descriptors (Schema v1)

Types and YAML values:
- SIFT: `type: "sift"`
- RGBSIFT: `type: "rgbsift"` (color; 3×128 stacking inside the descriptor). Set `use_color: true` if applicable.
- HoNC: `type: "honc"` (Histogram of Normalized Colors)
- vSIFT: `type: "vsift"` (Vanilla SIFT implementation)
- DSPSIFT: `type: "dspsift"` (Domain-Size Pooled SIFT; professor’s implementation)
- VGG: `type: "vgg"` (OpenCV xfeatures2d VGG descriptor; requires OpenCV contrib)

Notes
- DSPSIFT performs domain-size pooling internally by averaging descriptors across a range of measurement window sizes. To avoid double pooling, use `pooling: "none"` with DSPSIFT.
- To control the DSPSIFT pooling range via YAML, set `descriptors[].params.scales: [min, ..., max]`; the wrapper maps `[min, max]` to its internal range and uses count for the number of samples.
- VGG is intended as a non-pooled comparator; use `pooling: none` and L2 matching for fair baselines.

Example (DSPSIFT)
```yaml
descriptors:
  - name: "dspsift"
    type: "dspsift"
    pooling: "none"                # avoid double pooling
    scales: [0.85, 1.0, 1.3]       # maps to DSPSIFT linePoint1..linePoint2 and numScales
    normalize_after_pooling: true
```

## Profiling & Trade‑Off Metrics

The runner stores timing/throughput metrics in `results.metadata`:
- `detect_time_ms`: total detection time
- `compute_time_ms`: total descriptor extraction + pooling time
- `match_time_ms`: total matching time
- `total_images`, `total_keypoints`
- `images_per_sec`, `kps_per_sec`

Query latest (CSV):
```sql
SELECT r.processing_time_ms, r.metadata
FROM results r
ORDER BY r.id DESC
LIMIT 1;
```

See `docs/benchmarking.md` for more details and example queries.

## Designing Experiment Sets

- Naming conventions:
  - Place configs in `config/experiments/` with descriptive names (e.g., `sift_baseline.yaml`, `sift_dsp_gaussian.yaml`, `sift_rgbsift_stacking.yaml`).
  - Set `experiment.name` to a short slug per study (used in DB grouping), and use `descriptors[].name` to distinguish variants.

- Folder structure:
  - Keep templates in `config/defaults/` for reuse; copy into `config/experiments/` to run.
  - Results and DB records are stored under `results/` and `build/experiments.db` respectively.

- Scene selection:
  - Use `dataset.scenes: []` to run all scenes or provide a curated subset (e.g., `["i_ajuntament", "i_dome"]`).
  - Keep subsets small for iteration; run full-set for final reporting.

- Reproducibility:
  - For controlled comparisons, prefer `keypoints.source: homography_projection` and pre-generate locked keypoints with `keypoint_manager`.
  - For end-to-end evaluations, use `independent_detection`; ensure detector parameters are recorded.

- Versioning:
  - Include `experiment.version` and keep Schema v1 fields only; avoid deprecated keys.
  - Keep configs self-contained and link associated runs in your analysis notebooks.

## Troubleshooting

### Common Build Issues

#### OpenCV SIFT Not Found
```bash
# Verify SIFT support
python3 -c "import cv2; print('SIFT available:', hasattr(cv2, 'SIFT_create'))"

# If false, install OpenCV contrib:
# Manjaro: yay -S opencv-contrib
# Ubuntu: sudo apt install libopencv-contrib-dev
```

#### CMake Cache Issues
```bash
# Clean build directory
rm -rf build/
mkdir build && cd build
cmake .. && make
```

#### Docker Permission Issues
```bash
# Ensure user is in docker group
sudo usermod -aG docker $USER
# Log out and back in

# Fix file permissions
sudo chown -R $USER:$USER .
```

### Docker-Specific Issues

#### Container Won't Start
```bash
# Check Docker service
sudo systemctl status docker

# Rebuild image
docker-compose -f docker-compose.dev.yml build --no-cache
```

#### X11 Forwarding Issues
```bash
# Allow X11 connections
xhost +local:docker

# Test X11 in container
docker-compose -f docker-compose.dev.yml run descriptor-dev python3 -c "
import cv2
import numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
cv2.imshow('Test', img)
cv2.waitKey(1000)
"
```

#### Build Path Conflicts
```bash
# Use separate build directories
mkdir build-native    # for native builds
mkdir build-docker    # for Docker builds
```

### Dataset Issues

#### Dataset Not Found
```bash
# Verify dataset downloaded
ls data/
# Should show scene folders like i_ajuntament, v_wall, etc.

# Re-download if needed
python3 setup.py
```

#### Permission Issues
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/
```

### Windows-Specific Issues

#### Docker Desktop Requirements
```powershell
# Ensure WSL2 is enabled
wsl --install
# Restart computer after WSL2 installation

# Enable virtualization in BIOS if needed
# Docker Desktop → Settings → General → Use WSL2 based engine
```

#### Visual Studio Build Issues
```powershell
# Ensure correct Visual Studio components installed:
# - MSVC v143 compiler toolset
# - Windows 10/11 SDK
# - CMake tools for C++

# Set environment variables if using vcpkg
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
$env:CMAKE_TOOLCHAIN_FILE = "$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

#### Path Length Issues
```powershell
# Enable long path support (Windows 10+)
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### PowerShell Execution Policy
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Development Guidelines

### Adding New Descriptors
1. **Create descriptor class** inheriting from appropriate base
2. **Add to DescriptorType enum** in `experiment_config.hpp`
3. **Register in createDescriptorExtractor()** factory method
4. **Test with both native and Docker builds**

### Contributing
1. **Test changes** in both native and Docker environments
2. **Ensure reproducible results** across environments
3. **Update documentation** for new features
4. **Follow existing code structure** and naming conventions

### Performance Testing
- **Use Release builds** for performance measurements
- **Test in Docker** to ensure reproducible timing
- **Profile with tools** like gdb, valgrind (available in Docker)

## Quick Start Workflow

### 1. Setup
```bash
git clone <your-repository-url>
cd descriptor-compare
python3 setup.py                    # Download dataset
mkdir build && cd build
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DBUILD_DATABASE=ON
make -j$(nproc)
```

### 2. Setup Keypoints (Database-First)
```bash
# Generate keypoints directly to database (recommended)
./keypoint_manager generate ../data

# OR import existing CSV keypoints
./keypoint_manager import-csv ../reference_keypoints

# Verify keypoints loaded
./keypoint_manager list-scenes
```

### 3. Run Experiments (Database-Tracked)
```bash
# Run individual experiments (automatically tracked in database)
./experiment_runner ../config/experiments/sift_baseline.yaml
./experiment_runner ../config/experiments/rgbsift_comparison.yaml

# Check results in database
sqlite3 experiments.db "SELECT * FROM results ORDER BY timestamp DESC LIMIT 5;"

# OR check traditional CSV results
ls results/sift_baseline/sift/
cat results/sift_baseline/sift/i_dome/results.csv
```

### 4. Create Custom Experiments
```bash
# Copy existing config
cp ../config/experiments/sift_baseline.yaml ../config/experiments/my_experiment.yaml

# Edit your config
# Change experiment name, descriptors, parameters

# Run your experiment (results automatically tracked)
./experiment_runner ../config/experiments/my_experiment.yaml
```

### 5. Query and Analyze Results
```bash
# Query database for experiment comparison
sqlite3 experiments.db "
SELECT 
    e.descriptor_type,
    r.mean_average_precision,
    r.processing_time_ms 
FROM experiments e 
JOIN results r ON e.id = r.experiment_id 
ORDER BY r.mean_average_precision DESC;"

# Run analysis pipeline
./analysis_runner results/ --full

# View reports
open analysis/outputs/analysis_report.html
```

### 6. Export Results for Sharing
```bash
# Export keypoints for reproducibility
./keypoint_manager export-csv ./published_keypoints

# Share database file
cp experiments.db ./my_research_results.db
```

## Support

For issues, feature requests, or questions:
1. **Check troubleshooting section** above
2. **Verify environment setup** (native vs Docker)
3. **Test with minimal example** first
4. **Report with environment details** (OS, Docker version, etc.)

---

**Note**: This project requires the HPatches dataset for meaningful experiments. The `setup.py` script will automatically download and organize the dataset in the correct structure.
