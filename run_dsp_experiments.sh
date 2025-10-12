#!/bin/bash
#
# Run only the DSP-based descriptor experiments (SIFT DSP, RGB-SIFT DSP, HoNC DSP)
# Usage:
#   ./run_dsp_experiments.sh [group]
#
# Examples:
#   ./run_dsp_experiments.sh           # run all DSP experiment groups sequentially
#   ./run_dsp_experiments.sh sift      # run only the SIFT DSP group
#   ./run_dsp_experiments.sh parallel  # run all groups in parallel
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BUILD_DIR="build"
SPLIT_DSP_DIR="config/experiments/split_dsp"

# Map descriptors to config files
declare -A DSP_CONFIGS=(
    [sift]="$SPLIT_DSP_DIR/sift_dsp_experiments.yaml"
    [rgbsift]="$SPLIT_DSP_DIR/rgbsift_dsp_experiments.yaml"
    [honc]="$SPLIT_DSP_DIR/honc_dsp_experiments.yaml"
    [surf]="config/experiments/split/surf_experiments.yaml"
)

declare -A DSP_COUNTS=(
    [sift]="12 DSP variants"
    [rgbsift]="6 DSP variants"
    [honc]="6 DSP variants"
    [surf]="13 SURF variants"
)

if [ ! -f "$BUILD_DIR/experiment_runner" ]; then
    echo -e "${RED}Error: experiment_runner not found in $BUILD_DIR${NC}"
    echo "Please build the project first: cd build && cmake .. && make"
    exit 1
fi

run_group() {
    local key=$1
    local config=${DSP_CONFIGS[$key]}
    local count=${DSP_COUNTS[$key]}

    if [ ! -f "$config" ]; then
        echo -e "${RED}Missing config: $config${NC}"
        return 1
    fi

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running DSP group: ${key^^}${NC}"
    echo -e "${GREEN}Config: $config${NC}"
    echo -e "${GREEN}Descriptors: $count${NC}"
    echo -e "${GREEN}========================================${NC}"

    (cd "$BUILD_DIR" && ./experiment_runner "../$config")
}

show_usage() {
    cat <<USAGE
Usage: $0 [group|parallel]

DSP groups:
  sift     ${DSP_CONFIGS[sift]} (${DSP_COUNTS[sift]})
  rgbsift  ${DSP_CONFIGS[rgbsift]} (${DSP_COUNTS[rgbsift]})
  honc     ${DSP_CONFIGS[honc]} (${DSP_COUNTS[honc]})
  surf     ${DSP_CONFIGS[surf]} (${DSP_COUNTS[surf]})

Special modes:
  parallel run all groups concurrently
  help     display this message
USAGE
}

MODE=${1:-all}

case "$MODE" in
    sift|rgbsift|honc|surf)
        run_group "$MODE"
        ;;
    parallel)
        echo -e "${YELLOW}Running all DSP groups in parallel...${NC}"
        mkdir -p logs
        pids=()
        for key in sift rgbsift honc surf; do
            echo "Starting $key..."
            (run_group "$key" > "logs/${key}_dsp.log" 2>&1) &
            pids+=($!)
        done
        fail=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                ((fail++))
            fi
        done
        if [ $fail -eq 0 ]; then
            echo -e "${GREEN}All DSP groups completed successfully${NC}"
        else
            echo -e "${RED}$fail DSP group(s) failed${NC}"
            exit 1
        fi
        ;;
    all|"")
        echo -e "${YELLOW}Running all DSP experiment groups sequentially...${NC}"
        errors=()
        for key in sift rgbsift honc surf; do
            if ! run_group "$key"; then
                errors+=("$key")
            fi
            echo ""
        done
        if [ ${#errors[@]} -eq 0 ]; then
            echo -e "${GREEN}All DSP experiments finished successfully${NC}"
        else
            echo -e "${RED}Failures: ${errors[*]}${NC}"
            exit 1
        fi
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}Unknown option: $MODE${NC}"
        show_usage
        exit 1
        ;;
esac
