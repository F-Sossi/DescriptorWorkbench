#!/bin/bash
#
# Run all descriptor modification experiments + CNN baselines
# Splits experiments by descriptor type for easier management and parallel execution
#
# Usage:
#   ./run_full_experiments.sh [descriptor_type]
#
# Examples:
#   ./run_full_experiments.sh           # Run all experiments sequentially (SIFT, RGBSIFT, HoNC, CNN)
#   ./run_full_experiments.sh sift      # Run only SIFT modification experiments
#   ./run_full_experiments.sh cnn       # Run only CNN baseline experiments
#   ./run_full_experiments.sh parallel  # Run all experiments in parallel (background jobs)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build directory
BUILD_DIR="build"
SPLIT_DIR="config/experiments/split"
MAIN_DIR="config/experiments"

# Available experiment configs
declare -A EXPERIMENTS=(
    ["sift"]="$SPLIT_DIR/sift_experiments.yaml"
    ["rgbsift"]="$SPLIT_DIR/rgbsift_experiments.yaml"
    ["honc"]="$SPLIT_DIR/honc_experiments.yaml"
    ["cnn"]="$MAIN_DIR/cnn_descriptor_reference_keynet_sift_pairs.yaml"
)

declare -A DESCRIPTOR_COUNTS=(
    ["sift"]="36 modifications"
    ["rgbsift"]="12 modifications"
    ["honc"]="18 modifications"
    ["cnn"]="3 baselines (HardNet, SOSNet, L2Net)"
)

# Check if experiment_runner exists
if [ ! -f "$BUILD_DIR/experiment_runner" ]; then
    echo -e "${RED}Error: experiment_runner not found in $BUILD_DIR${NC}"
    echo "Please build the project first: cd build && cmake .. && make"
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local desc_type=$1
    local config_file=$2
    local desc_count=$3

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running ${desc_type^^} experiments${NC}"
    echo -e "${GREEN}Config: $config_file${NC}"
    echo -e "${GREEN}Descriptors: $desc_count${NC}"
    echo -e "${GREEN}========================================${NC}"

    cd "$BUILD_DIR"

    # Run experiment
    if ./experiment_runner "../$config_file"; then
        echo -e "${GREEN}✓ ${desc_type^^} experiments completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ ${desc_type^^} experiments failed${NC}"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [descriptor_type|parallel]"
    echo ""
    echo "Available descriptor types:"
    for desc_type in "${!EXPERIMENTS[@]}"; do
        config="${EXPERIMENTS[$desc_type]}"
        count="${DESCRIPTOR_COUNTS[$desc_type]}"
        printf "  %-10s %s (%s descriptors)\n" "$desc_type" "$config" "$count"
    done
    echo ""
    echo "Special modes:"
    echo "  parallel   Run all experiments in parallel (background jobs)"
    echo "  all        Run all experiments sequentially (default)"
}

# Parse command line arguments
MODE="${1:-all}"

case "$MODE" in
    sift|rgbsift|honc|cnn)
        # Run single experiment
        config="${EXPERIMENTS[$MODE]}"
        count="${DESCRIPTOR_COUNTS[$MODE]}"
        if [ -z "$config" ]; then
            echo -e "${RED}Unknown descriptor type: $MODE${NC}"
            show_usage
            exit 1
        fi
        run_experiment "$MODE" "$config" "$count"
        ;;

    parallel)
        # Run all experiments in parallel
        echo -e "${YELLOW}Starting all experiments in parallel...${NC}"
        echo -e "${YELLOW}WARNING: This will consume significant CPU resources${NC}"
        echo ""

        pids=()
        for desc_type in sift rgbsift honc cnn; do
            config="${EXPERIMENTS[$desc_type]}"
            count="${DESCRIPTOR_COUNTS[$desc_type]}"

            echo "Starting $desc_type in background..."
            (run_experiment "$desc_type" "$config" "$count" > "logs/${desc_type}_experiments.log" 2>&1) &
            pids+=($!)
        done

        # Wait for all background jobs
        echo ""
        echo "Waiting for all experiments to complete..."
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                ((failed++))
            fi
        done

        if [ $failed -eq 0 ]; then
            echo -e "${GREEN}All parallel experiments completed successfully${NC}"
        else
            echo -e "${RED}$failed experiment(s) failed${NC}"
            exit 1
        fi
        ;;

    all|"")
        # Run all experiments sequentially
        echo -e "${YELLOW}Running all descriptor experiments + CNN baselines sequentially...${NC}"
        echo ""

        failed_experiments=()
        for desc_type in sift rgbsift honc cnn; do
            config="${EXPERIMENTS[$desc_type]}"
            count="${DESCRIPTOR_COUNTS[$desc_type]}"

            if ! run_experiment "$desc_type" "$config" "$count"; then
                failed_experiments+=("$desc_type")
            fi
            echo ""
        done

        # Summary
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}SUMMARY${NC}"
        echo -e "${GREEN}========================================${NC}"

        if [ ${#failed_experiments[@]} -eq 0 ]; then
            echo -e "${GREEN}All experiments completed successfully!${NC}"
            echo ""
            echo "Experiments run:"
            echo "  - SIFT modifications (36 descriptors)"
            echo "  - RGBSIFT modifications (12 descriptors)"
            echo "  - HoNC modifications (18 descriptors)"
            echo "  - CNN baselines (3 descriptors: HardNet, SOSNet, L2Net)"
            echo ""
            echo "Total: 69 descriptor configurations tested"
        else
            echo -e "${RED}Failed experiments: ${failed_experiments[*]}${NC}"
            exit 1
        fi
        ;;

    help|--help|-h)
        show_usage
        ;;

    *)
        echo -e "${RED}Unknown option: $MODE${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
