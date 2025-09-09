#!/bin/bash

# Descriptor Analysis Launcher Script
echo "🔬 DescriptorWorkbench Analysis Environment"
echo "=========================================="

# Activate conda environment
echo "📦 Activating conda environment..."
source /home/frank/miniforge3/etc/profile.d/conda.sh
conda activate descriptor-compare

# Check database status
echo "🗄️  Checking database status..."
DB_PATH="build/experiments.db"
if [ -f "$DB_PATH" ]; then
    EXPERIMENT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM experiments;")
    RESULT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM results;")
    echo "   ✅ Database found: $EXPERIMENT_COUNT experiments, $RESULT_COUNT results"
else
    echo "   ⚠️  Database not found. Run experiments first!"
    echo "   Run: cd build && ./experiment_runner ../config/experiments/sift_systematic_analysis.yaml"
    exit 1
fi

# Check keypoint status
echo "🔍 Checking keypoint status..."
KEYPOINT_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM locked_keypoints;")
KEYPOINT_SETS=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM keypoint_sets;")
echo "   ✅ Keypoints: $KEYPOINT_COUNT keypoints in $KEYPOINT_SETS sets"

echo ""
echo "🚀 Starting JupyterLab..."
echo "📊 Analysis notebooks available:"
echo "   • 01_descriptor_performance_overview.ipynb"
echo "   • 02_pooling_stacking_effects.ipynb" 
echo "   • 03_systematic_modification_analysis.ipynb"
echo ""
echo "🌐 JupyterLab will be available at: http://localhost:8888"
echo "📁 Navigate to: analysis/notebooks/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="

# Launch JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser