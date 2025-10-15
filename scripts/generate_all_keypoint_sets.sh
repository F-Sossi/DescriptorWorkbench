#!/bin/bash
# Script to generate all required keypoint sets for the DescriptorWorkbench experiment suite
# This script creates keypoint sets for detector-specific and spatial intersection comparisons

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
DATA_DIR="$PROJECT_ROOT/data"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Keypoint Set Generation Script"
echo "========================================="
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found at $BUILD_DIR${NC}"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF -DBUILD_DATABASE=ON"
    echo "  make -j\$(nproc)"
    exit 1
fi

# Check if keypoint_manager exists
if [ ! -f "$BUILD_DIR/keypoint_manager" ]; then
    echo -e "${RED}Error: keypoint_manager executable not found${NC}"
    echo "Please build the CLI tools first"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found at $DATA_DIR${NC}"
    echo "Please download the HPatches dataset first:"
    echo "  python3 setup.py"
    exit 1
fi

cd "$BUILD_DIR"

echo "Generating keypoint sets..."
echo ""

# ============================================
# 1. Pure Detector Keypoint Sets
# ============================================

echo -e "${GREEN}[1/7] Generating SIFT keypoints (sift_keypoints)${NC}"
./keypoint_manager generate-detector "$DATA_DIR" sift sift_keypoints
echo ""

echo -e "${GREEN}[2/7] Generating ORB keypoints (orb_keypoints)${NC}"
./keypoint_manager generate-detector "$DATA_DIR" orb orb_keypoints
echo ""

echo -e "${GREEN}[3/7] Generating SURF keypoints (surf_keypoints)${NC}"
./keypoint_manager generate-detector "$DATA_DIR" surf surf_keypoints
echo ""

# ============================================
# 2. Non-Overlapping Keypoints for CNN Comparison
# ============================================

echo -e "${GREEN}[4/7] Generating non-overlapping SIFT keypoints for CNN comparison (sift_keynet_pairs, 32px min distance)${NC}"
./keypoint_manager generate-non-overlapping "$DATA_DIR" sift 32.0 sift_keynet_pairs
echo ""

# ============================================
# 3. Spatial Intersection Keypoint Sets (5px tolerance)
# ============================================

echo -e "${GREEN}[5/7] Generating ORB-SIFT spatial intersection keypoints (orb_sift_intersection, 5px)${NC}"
echo -e "${YELLOW}Note: This requires implementation of spatial intersection in keypoint_manager${NC}"
echo -e "${YELLOW}TODO: Add spatial intersection support to CLI tool${NC}"
echo ""

echo -e "${GREEN}[6/7] Generating SURF-SIFT spatial intersection keypoints (surf_sift_intersection, 5px)${NC}"
echo -e "${YELLOW}Note: This requires implementation of spatial intersection in keypoint_manager${NC}"
echo -e "${YELLOW}TODO: Add spatial intersection support to CLI tool${NC}"
echo ""

# ============================================
# 4. Optional: Harris Corner Detection
# ============================================

echo -e "${GREEN}[7/7] Generating Harris corner keypoints (harris_keypoints)${NC}"
./keypoint_manager generate-detector "$DATA_DIR" harris harris_keypoints
echo ""

# ============================================
# Summary
# ============================================

echo ""
echo "========================================="
echo "Keypoint Set Generation Summary"
echo "========================================="
echo ""
echo "✅ Generated keypoint sets:"
echo "  - sift_keypoints (pure SIFT detection)"
echo "  - orb_keypoints (pure ORB detection)"
echo "  - surf_keypoints (pure SURF detection)"
echo "  - sift_keynet_pairs (non-overlapping for CNN, 32px min distance)"
echo "  - harris_keypoints (Harris corner detection)"
echo ""
echo "⚠️  Spatial intersection sets require CLI tool update:"
echo "  - orb_sift_intersection (5px spatial matching)"
echo "  - surf_sift_intersection (5px spatial matching)"
echo ""
echo "To verify keypoint sets in database:"
echo "  cd $BUILD_DIR"
echo "  ./keypoint_manager list-scenes"
echo ""
echo "Database location: $BUILD_DIR/experiments.db"
echo ""
