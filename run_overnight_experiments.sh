#!/bin/bash

# Comprehensive Overnight Descriptor Analysis
# Runs systematic analysis for all descriptor types
# 
# IMPORTANT: Ensure you have enough disk space and the database is backed up
# Total estimated time: 8-12 hours depending on hardware
# Expected experiments: ~41 descriptor configurations

echo "=========================================="
echo "OVERNIGHT DESCRIPTOR ANALYSIS STARTING"
echo "=========================================="

# Resolve repository root (directory of this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BUILD_DIR="$REPO_ROOT/build"
CONFIG_DIR="$REPO_ROOT/config/experiments"
# IMPORTANT: Binaries expect DB and relative paths from CWD=build
DB_FILE="$BUILD_DIR/experiments.db"

# Create logs directory 
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Get timestamp for this batch
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="$LOG_DIR/overnight_batch_$BATCH_TIMESTAMP.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG"
}

# Function to run experiment with error handling
run_experiment() {
    local config_file="$1"        # absolute path to YAML
    local experiment_name
    experiment_name=$(basename "$config_file" .yaml)
    local exp_log="$LOG_DIR/${experiment_name}_$BATCH_TIMESTAMP.log"

    log " Starting experiment: $experiment_name"
    echo "======================================" >> "$exp_log"
    echo "EXPERIMENT: $experiment_name" >> "$exp_log"
    echo "CONFIG: $config_file" >> "$exp_log"
    echo "START TIME: $(date)" >> "$exp_log"
    echo "======================================" >> "$exp_log"

    # Run from build directory so relative paths in YAML/DB align
    if (cd "$BUILD_DIR" && ./experiment_runner "$config_file") >> "$exp_log" 2>&1; then
        log "✅ Completed experiment: $experiment_name"
        echo "SUCCESS: $(date)" >> "$exp_log"
        return 0
    else
        log "❌ Failed experiment: $experiment_name (check $exp_log)"
        echo "FAILED: $(date)" >> "$exp_log"
        return 1
    fi
}

# Function to show database stats
show_db_stats() {
    if [ -f "$DB_FILE" ]; then
        local exp_count=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM experiments;" 2>/dev/null || echo "0")
        local result_count=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM results;" 2>/dev/null || echo "0")
        log "📊 Database stats: $exp_count experiments, $result_count results"
    else
        log "📊 Database not found - will be created"
    fi
}

# Pre-flight checks
log "🔍 Pre-flight checks..."

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    log "❌ Build directory not found: $BUILD_DIR"
    exit 1
fi

# Check if experiment_runner exists
if [ ! -x "$BUILD_DIR/experiment_runner" ]; then
    log "❌ experiment_runner not found or not executable: $BUILD_DIR/experiment_runner"
    log " Please build: cmake -S $REPO_ROOT -B $BUILD_DIR && cmake --build $BUILD_DIR -j\$(nproc)"
    exit 1
fi

# Check if keypoint_manager exists (used for keypoint generation)
if [ ! -x "$BUILD_DIR/keypoint_manager" ]; then
    log "❌ keypoint_manager not found or not executable: $BUILD_DIR/keypoint_manager"
    log " Please build: cmake -S $REPO_ROOT -B $BUILD_DIR && cmake --build $BUILD_DIR -j\$(nproc)"
    exit 1
fi

# Check disk space (require at least 2GB free)
AVAILABLE_SPACE=$(df "$BUILD_DIR" | awk 'NR==2{print $4}')
if [ "$AVAILABLE_SPACE" -lt 2000000 ]; then
    log " Low disk space: $(($AVAILABLE_SPACE/1024))MB available"
    log "Recommend at least 2GB free space"
fi

# Show initial database state
show_db_stats

# Backup existing database if it exists
if [ -f "$DB_FILE" ]; then
    BACKUP_FILE="${DB_FILE}.backup_$BATCH_TIMESTAMP"
    cp "$DB_FILE" "$BACKUP_FILE"
    log " Created database backup: $BACKUP_FILE"
fi

# Generate keypoint sets for experiments
log " Generating keypoint sets..."

# Check if keypoint sets already exist
KEYPOINT_COUNT=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM locked_keypoints;" 2>/dev/null || echo "0")
if [ "$KEYPOINT_COUNT" -gt 0 ]; then
    log " Found $KEYPOINT_COUNT existing keypoints in database"
    log " Skipping keypoint generation (use 'rm $DB_FILE' to regenerate)"
else
    log " Generating homography projection keypoints (controlled evaluation)..."
    (cd "$BUILD_DIR" && ./keypoint_manager generate-projected ../data "homography_projection_systematic") >> "$BATCH_LOG" 2>&1
    
    # Check if keypoints were actually generated
    HOMOG_KEYPOINTS=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM locked_keypoints WHERE keypoint_set_id = (SELECT id FROM keypoint_sets WHERE name = 'homography_projection_systematic');" 2>/dev/null || echo "0")
    if [ "$HOMOG_KEYPOINTS" -gt 0 ]; then
        log "✅ Generated $HOMOG_KEYPOINTS homography projection keypoints"
    else
        log "❌ Failed to generate homography projection keypoints (check $BATCH_LOG)"
        exit 1
    fi
    
    log " Generating independent detection keypoints (realistic evaluation)..."
    (cd "$BUILD_DIR" && ./keypoint_manager generate-independent ../data "independent_detection_systematic") >> "$BATCH_LOG" 2>&1
    
    # Check if keypoints were actually generated
    INDEP_KEYPOINTS=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM locked_keypoints WHERE keypoint_set_id = (SELECT id FROM keypoint_sets WHERE name = 'independent_detection_systematic');" 2>/dev/null || echo "0")
    if [ "$INDEP_KEYPOINTS" -gt 0 ]; then
        log "✅ Generated $INDEP_KEYPOINTS independent detection keypoints"
    else
        log "❌ Failed to generate independent detection keypoints (check $BATCH_LOG)"
        exit 1
    fi
    
    # Show keypoint generation results
    NEW_KEYPOINT_COUNT=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM locked_keypoints;" 2>/dev/null || echo "0")
    log "📊 Generated $NEW_KEYPOINT_COUNT total keypoints across both methods"
fi

log "🏁 All checks passed - starting experiments..."

# Track experiment results
TOTAL_EXPERIMENTS=0
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# List of all systematic analysis configs
EXPERIMENT_CONFIGS=(
    "$CONFIG_DIR/sift_systematic_analysis.yaml"
    "$CONFIG_DIR/honc_systematic_analysis.yaml"
    "$CONFIG_DIR/vsift_systematic_analysis.yaml"
    "$CONFIG_DIR/vgg_systematic_analysis.yaml"
    "$CONFIG_DIR/dspsift_systematic_analysis.yaml"
)

# Run each experiment set
for config in "${EXPERIMENT_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
        if run_experiment "$config"; then
            SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
        else
            FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
        fi

        # Show progress
        log "📈 Progress: $SUCCESSFUL_EXPERIMENTS successful, $FAILED_EXPERIMENTS failed"

        # Brief pause between experiments
        sleep 5
    else
        log "Error:  Config file not found: $config"
    fi
done

# Final statistics
log "=========================================="
log " OVERNIGHT ANALYSIS COMPLETE!"
log "=========================================="
log "📊 Final Results:"
log "   • Total experiment sets: $TOTAL_EXPERIMENTS"
log "   • Successful: $SUCCESSFUL_EXPERIMENTS"
log "   • Failed: $FAILED_EXPERIMENTS"
if [ "$TOTAL_EXPERIMENTS" -gt 0 ]; then
    log "   • Success rate: $(($SUCCESSFUL_EXPERIMENTS * 100 / $TOTAL_EXPERIMENTS))%"
else
    log "   • Success rate: N/A (no experiments found)"
fi

# Show final database state
show_db_stats

# Show top performing results
if [ -f "$DB_FILE" ]; then
    log ""
    log "🏆 Top 5 performing configurations:"
    sqlite3 "$DB_FILE" "
    SELECT 
        e.descriptor_type,
        printf('%.4f', r.true_map_macro) as macro_map,
        printf('%.1f', r.processing_time_ms/1000) as time_sec
    FROM experiments e 
    JOIN results r ON e.id = r.experiment_id 
    WHERE r.true_map_macro IS NOT NULL
    ORDER BY r.true_map_macro DESC 
    LIMIT 5;" 2>/dev/null | while read line; do
        log "   $line"
    done
fi

# Cleanup and recommendations
log ""
log " Log files saved to: $LOG_DIR"
log " Database location: $DB_FILE"
if [ -f "$BACKUP_FILE" ]; then
    log "🔄 Backup available: $BACKUP_FILE"
fi

log ""
log "🔬 Next steps:"
log "   1. Analyze results: jupyter lab analysis/notebooks/"
log "   2. View database: sqlite3 $DB_FILE"
log "   3. Generate reports: ./analysis_runner"

if [ $FAILED_EXPERIMENTS -gt 0 ]; then
    log ""
    log "  Some experiments failed - check individual logs in $LOG_DIR"
    exit 1
else
    log ""
    log "All experiments completed successfully!"
    exit 0
fi
