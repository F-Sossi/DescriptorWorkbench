#!/bin/bash
#
# Run every curated DescriptorWorkbench experiment configuration sequentially.
# Logs each run to logs/<basename>.log and stops only after attempting all configs.
#
# Usage:
#   ./run_all_experiments.sh
#

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
RUNNER="$BUILD_DIR/experiment_runner"
LOG_DIR="$ROOT_DIR/logs"

if [ ! -x "$RUNNER" ]; then
    echo "experiment_runner not found in $BUILD_DIR"
    echo "Build the project first:"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -DUSE_SYSTEM_PACKAGES=ON"
    echo "  cmake --build ."
    exit 1
fi

mkdir -p "$LOG_DIR"

CONFIGS=(
"config/experiments/dsprgbsift_intersection.yaml"
"config/experiments/scale_vs_intersection_study.yaml"
)

failures=()
for yaml in "${CONFIGS[@]}"; do
    name="$(basename "$yaml" .yaml)"
    log_file="$LOG_DIR/${name}.log"

    echo "========================================="
    echo "Running $yaml"
    echo "Log: $log_file"
    echo "========================================="

    if (cd "$BUILD_DIR" && ./experiment_runner "../$yaml" | tee "$log_file"); then
        echo "✓ $yaml completed"
    else
        echo "✗ $yaml failed (see $log_file)"
        failures+=("$yaml")
    fi

    echo ""
done

if [ ${#failures[@]} -eq 0 ]; then
    echo "All experiments finished successfully."
else
    echo "The following experiments failed:"
    for f in "${failures[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
