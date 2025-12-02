#!/bin/bash
#
# Generate all required keypoint sets for DescriptorWorkbench experiments
#
# Usage: ./generate_keypoint_sets.sh
#
# This script can be run from the root directory or build/ directory
#

set -euo pipefail

echo "==========================================="
echo "Keypoint Set Generation Script"
echo "==========================================="
echo ""

# Get script directory and repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR"

# Change to build directory
BUILD_DIR="$REPO_ROOT/build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory not found at $BUILD_DIR"
    echo "Please run cmake and make first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON"
    echo "  make -j\$(nproc)"
    exit 1
fi

cd "$BUILD_DIR"
echo "Working directory: $BUILD_DIR"

# Check keypoint_manager exists
if [ ! -f "./keypoint_manager" ]; then
    echo "ERROR: keypoint_manager not found in $BUILD_DIR"
    echo "Please build the project first:"
    echo "  cd build"
    echo "  cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON"
    echo "  make -j\$(nproc)"
    exit 1
fi

# Check data directory exists
DATA_DIR="$REPO_ROOT/data"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    echo "Please ensure HPatches dataset is downloaded to data/"
    exit 1
fi

echo "Data directory: $DATA_DIR"
echo ""

# ============================================================================
# Helper function to check if keypoint set exists
# ============================================================================

set_exists() {
    local set_name="$1"
    ./keypoint_manager list-sets 2>/dev/null | grep -q "^ID [0-9]*: $set_name"
    return $?
}

# ============================================================================
# STEP 1: Base keypoint sets (native detectors)
# ============================================================================

echo "==========================================="
echo "STEP 1: Generating Base Keypoint Sets"
echo "==========================================="
echo ""

echo "[1/3] Generating sift_8000..."
if set_exists "sift_8000"; then
    echo "✓ sift_8000 already exists, skipping..."
else
    ./keypoint_manager generate-detector "$DATA_DIR" sift sift_8000 --max-features 8000
fi
echo ""

echo "[2/3] Generating surf_8000..."
if set_exists "surf_8000"; then
    echo "✓ surf_8000 already exists, skipping..."
else
    ./keypoint_manager generate-detector "$DATA_DIR" surf surf_8000 --max-features 8000
fi
echo ""

echo "[3/3] Generating keynet_8000..."
if set_exists "keynet_8000"; then
    echo "✓ keynet_8000 already exists, skipping..."
else
    ./keypoint_manager generate-kornia-keynet "$DATA_DIR" keynet_8000 --max-features 8000 --mode independent
fi
echo ""

echo "Base keypoint sets complete!"
echo ""

# ============================================================================
# STEP 2: Scale-controlled keypoint sets (top 25% by size)
# ============================================================================

echo "==========================================="
echo "STEP 2: Generating Scale-Controlled Sets"
echo "==========================================="
echo ""

echo "[1/3] Generating sift_scale_matched_6px (top 645k by size)..."
if set_exists "sift_scale_matched_6px"; then
    echo "✓ sift_scale_matched_6px already exists, skipping..."
else
    ./keypoint_manager generate-top-n sift_8000 645000 --by size --out sift_scale_matched_6px
fi
echo ""

echo "[2/3] Generating surf_scale_matched_6px (top 645k by size)..."
if set_exists "surf_scale_matched_6px"; then
    echo "✓ surf_scale_matched_6px already exists, skipping..."
else
    ./keypoint_manager generate-top-n surf_8000 645000 --by size --out surf_scale_matched_6px
fi
echo ""

echo "[3/3] Generating keynet_scale_matched_6px (top 645k by size)..."
if set_exists "keynet_scale_matched_6px"; then
    echo "✓ keynet_scale_matched_6px already exists, skipping..."
else
    ./keypoint_manager generate-top-n keynet_8000 645000 --by size --out keynet_scale_matched_6px
fi
echo ""

echo "Scale-controlled sets complete!"
echo ""

# ============================================================================
# STEP 3: SIFT-SURF intersection sets
# ============================================================================

echo "==========================================="
echo "STEP 3: Generating SIFT-SURF Intersections"
echo "==========================================="
echo ""

echo "[1/2] Generating sift_surf_8k_a and sift_surf_8k_b (3px tolerance)..."
if set_exists "sift_surf_8k_a" && set_exists "sift_surf_8k_b"; then
    echo "✓ sift_surf_8k_a and sift_surf_8k_b already exist, skipping..."
else
    ./keypoint_manager build-intersection \
        --source-a sift_8000 \
        --source-b surf_8000 \
        --out-a sift_surf_8k_a \
        --out-b sift_surf_8k_b \
        --tolerance 3.0 \
        --overwrite
fi
echo ""

echo "[2/2] Generating sift_surf_scale_matched_intersection_a and b (6px tolerance, from scale-controlled)..."
if set_exists "sift_surf_scale_matched_intersection_a" && set_exists "sift_surf_scale_matched_intersection_b"; then
    echo "✓ sift_surf_scale_matched_intersection_a and b already exist, skipping..."
else
    ./keypoint_manager build-intersection \
        --source-a sift_scale_matched_6px \
        --source-b surf_scale_matched_6px \
        --out-a sift_surf_scale_matched_intersection_a \
        --out-b sift_surf_scale_matched_intersection_b \
        --tolerance 6.0 \
        --overwrite
fi
echo ""

echo "SIFT-SURF intersections complete!"
echo ""

# ============================================================================
# STEP 4: KeyNet-SIFT intersection sets
# ============================================================================

echo "==========================================="
echo "STEP 4: Generating KeyNet-SIFT Intersections"
echo "==========================================="
echo ""

echo "[1/2] Generating keynet_sift_8k_a and keynet_sift_8k_b (5px tolerance)..."
if set_exists "keynet_sift_8k_a" && set_exists "keynet_sift_8k_b"; then
    echo "✓ keynet_sift_8k_a and keynet_sift_8k_b already exist, skipping..."
else
    ./keypoint_manager build-intersection \
        --source-a keynet_8000 \
        --source-b sift_8000 \
        --out-a keynet_sift_8k_a \
        --out-b keynet_sift_8k_b \
        --tolerance 5.0 \
        --overwrite
fi
echo ""

echo "[2/2] Generating sift_keynet_scale_matched_intersection_a and b (5px tolerance, from scale-controlled)..."
if set_exists "sift_keynet_scale_matched_intersection_a" && set_exists "sift_keynet_scale_matched_intersection_b"; then
    echo "✓ sift_keynet_scale_matched_intersection_a and b already exist, skipping..."
else
    ./keypoint_manager build-intersection \
        --source-a sift_scale_matched_6px \
        --source-b keynet_scale_matched_6px \
        --out-a sift_keynet_scale_matched_intersection_a \
        --out-b sift_keynet_scale_matched_intersection_b \
        --tolerance 5.0 \
        --overwrite
fi
echo ""

echo "KeyNet-SIFT intersections complete!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "==========================================="
echo "Summary: All Keypoint Sets Generated"
echo "==========================================="
echo ""

echo "Listing all keypoint sets:"
./keypoint_manager list-sets
echo ""

echo "==========================================="
echo "SUCCESS: All keypoint sets ready!"
echo "==========================================="
echo ""
echo "You can now run experiments with:"
echo "  cd $BUILD_DIR"
echo "  ./experiment_runner ../config/experiments/<config>.yaml"
echo ""
