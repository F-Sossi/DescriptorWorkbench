# Stage 5: Database Integration

## Overview
Stage 5 adds experiment tracking capabilities without disrupting the workflow. Database integration is now required and always built into the project.

## Key Features

### 1. Integration
- Database tracking is always compiled in
- CLI runs automatically persist experiments and metrics

### 2. CLI-First Workflow
- Use `experiment_runner` and `keypoint_manager` CLIs
- `descriptor_compare` components remain as libraries used internally

### 3. SQLite-Based Storage
- Lightweight, embedded database
- No external database server required
- Human-readable data storage

## Usage

### Basic Usage
```bash
# Configure (database integration is always enabled)
cmake .. -DUSE_SYSTEM_PACKAGES=ON -DUSE_CONAN=OFF
make

# Run experiments (results recorded automatically)
./experiment_runner ../config/experiments/sift_baseline.yaml

# Test database functionality
ctest -R database --output-on-failure
```

### Code Integration Example
```cpp
#include "thesis_project/database/DatabaseManager.hpp"

// In your existing image_processor workflow:
void processExperiments() {
    // Database manager (set enabled=false only if you explicitly want to skip persistence)
    thesis_project::database::DatabaseManager db("experiments.db", true);

    // Your existing experiment code unchanged
    experiment_config config;
    // ... existing setup ...

    // Record experiment when database persistence is active
    if (db.isEnabled()) {
        auto db_config = database_integration::toDbConfig(config);
        int exp_id = db.recordConfiguration(db_config);

        // After processing...
        auto results = database_integration::createDbResults(
            exp_id, "RGBSIFT", "i_ajuntament", map_score, time_ms);
        db.recordExperiment(results);
    }
}
```

## Database Schema

### Experiments Table
- `id`: Primary key
- `descriptor_type`: SIFT, RGBSIFT, etc.
- `dataset_name`: Dataset identifier
- `pooling_strategy`: NONE, STACKING, etc.
- `similarity_threshold`: Matching threshold
- `max_features`: Feature limit
- `timestamp`: When experiment was run
- `parameters`: Additional parameters

### Results Table
- `id`: Primary key
- `experiment_id`: Foreign key to experiments
- `mean_average_precision`: MAP score
- `precision_at_1`, `precision_at_5`: Precision metrics
- `recall_at_1`, `recall_at_5`: Recall metrics
- `total_matches`: Number of matches found
- `total_keypoints`: Total keypoints detected
- `processing_time_ms`: Processing duration
- `timestamp`: When results were recorded
- `metadata`: Additional result metadata

## API Reference

### DatabaseManager Class
```cpp
class DatabaseManager {
public:
    DatabaseManager(const std::string& db_path, bool enabled = false);

    bool isEnabled() const;
    int recordConfiguration(const ExperimentConfig& config);
    bool recordExperiment(const ExperimentResults& results);
    std::vector<ExperimentResults> getRecentResults(int limit = 10);
    std::map<std::string, double> getStatistics();
};
```

### Helper Functions
```cpp
// Convert existing config to database format
auto db_config = database_integration::toDbConfig(experiment_config);

// Create results structure
auto results = database_integration::createDbResults(
    exp_id, descriptor_name, dataset_name, map_score, time_ms);
```

## Benefits

1. **Non-Disruptive**: Existing workflow unchanged
2. **Persistent**: Every run is captured without extra flags
3. **Lightweight**: SQLite embedded database
4. **Automatic**: Tracks experiments transparently
5. **Analyzable**: Provides statistics and history
6. **Flexible**: Easy to extend with new metrics

## Adoption

1. **Immediate**: Use CLIs; database is on by default
2. **Config**: Add experiment YAMLs under `config/experiments/`
3. **Extend**: Track additional metrics via DatabaseManager
4. **Analysis**: Use Stage 6 analysis tools

## Testing

Run the validation script to test all functionality:
```bash
./validate_stage5.sh
```

This tests:
- Database creation and initialization
- Experiment recording and retrieval
- Statistics generation
- Build system integration
- CLI usage validated
