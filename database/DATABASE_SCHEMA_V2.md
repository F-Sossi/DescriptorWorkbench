# Database Schema Documentation v2.0

## Overview

The DescriptorWorkbench database stores experiment configurations, results, and keypoint data for computer vision descriptor research. This document describes the v2.0 schema which introduces true IR-style mAP metrics as first-class database columns.

## Schema Version History

- **v1.0**: Initial schema with basic experiment tracking
- **v2.0 (September 2025)**: **Major upgrade** - True IR-style mAP metrics promoted from metadata strings to primary columns

## Migration from v1.0 to v2.0

Use the provided migration script to upgrade existing databases:

```bash
cd database/
python3 migrate_database.py ../build/experiments.db
```

The migration:
- Adds 4 new true MAP columns to the `results` table
- Extracts values from metadata strings using regex
- Preserves all existing data with backward compatibility
- Creates automatic backup before migration

## Table Schemas

### experiments

Stores experiment configurations and parameters.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique experiment identifier |
| `descriptor_type` | TEXT NOT NULL | Descriptor name (e.g., "sift_grayscale_dsp_l2") |
| `dataset_name` | TEXT NOT NULL | Dataset path (e.g., "../data/") |
| `pooling_strategy` | TEXT | Pooling method: "none", "domain_size_pooling", "stacking" |
| `similarity_threshold` | REAL | Matching threshold (typically 0.8) |
| `max_features` | INTEGER | Maximum keypoints per image |
| `timestamp` | TEXT NOT NULL | Experiment execution time |
| `parameters` | TEXT | Additional configuration (JSON format) |

### results  **UPDATED in v2.0**

Stores experiment results with IR-style mAP metrics as primary columns.

#### Primary IR-style mAP Metrics (NEW in v2.0)
| Column | Type | Description |
|--------|------|-------------|
| `true_map_macro` | REAL | **Scene-balanced mAP** - Primary metric for evaluation |
| `true_map_micro` | REAL | **Overall mAP** - Weighted by total query count |
| `true_map_macro_with_zeros` | REAL | **Conservative macro** - Includes R=0 queries as AP=0 |
| `true_map_micro_with_zeros` | REAL | **Conservative micro** - Includes R=0 queries as AP=0 |

#### Legacy/Compatibility Metrics
| Column | Type | Description |
|--------|------|-------------|
| `mean_average_precision` | REAL | **Primary display metric** - Uses `true_map_macro` when available |
| `legacy_mean_precision` | REAL | **Original arithmetic mean** - For backward compatibility |

#### Standard Retrieval Metrics
| Column | Type | Description |
|--------|------|-------------|
| `precision_at_1` | REAL | Precision at rank 1 (exact match rate) |
| `precision_at_5` | REAL | Precision at rank 5 |
| `recall_at_1` | REAL | Recall at rank 1 |
| `recall_at_5` | REAL | Recall at rank 5 |

#### Experiment Metadata
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique result identifier |
| `experiment_id` | INTEGER FK | Links to `experiments.id` |
| `total_matches` | INTEGER | Total descriptor matches found |
| `total_keypoints` | INTEGER | Total keypoints processed |
| `processing_time_ms` | REAL | Total experiment time (milliseconds) |
| `timestamp` | TEXT NOT NULL | Result recording time |
| `metadata` | TEXT | Additional metrics and profiling data (JSON) |

### keypoint_sets

Manages different keypoint generation strategies for controlled experiments.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique keypoint set identifier |
| `name` | TEXT UNIQUE | Set name (e.g., "homography_projection_default") |
| `generator_type` | TEXT | Keypoint detector: "SIFT", "ORB", "AKAZE" |
| `generation_method` | TEXT | Strategy: "homography_projection", "independent_detection" |
| `max_features` | INTEGER | Maximum keypoints per image |
| `dataset_path` | TEXT | Source dataset path |
| `description` | TEXT | Human-readable description |
| `boundary_filter_px` | INTEGER | Boundary filter distance (default: 40px) |
| `created_at` | TIMESTAMP | Set creation time |

### locked_keypoints

Stores pre-computed keypoints for consistent evaluation across experiments.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique keypoint identifier |
| `keypoint_set_id` | INTEGER FK | Links to `keypoint_sets.id` |
| `scene_name` | TEXT | Scene identifier (e.g., "i_dome", "v_wall") |
| `image_name` | TEXT | Image identifier (e.g., "1.ppm") |
| `x` | REAL | Keypoint x-coordinate |
| `y` | REAL | Keypoint y-coordinate |
| `size` | REAL | Keypoint scale |
| `angle` | REAL | Keypoint orientation (degrees) |
| `response` | REAL | Detector response strength |
| `octave` | INTEGER | Scale octave |
| `class_id` | INTEGER | Keypoint class (typically -1) |
| `valid_bounds` | BOOLEAN | Within image boundaries (default: 1) |
| `created_at` | TIMESTAMP | Keypoint storage time |

### descriptors

Stores computed descriptors for research analysis (optional table).

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Unique descriptor identifier |
| `experiment_id` | INTEGER FK | Links to `experiments.id` |
| `scene_name` | TEXT | Scene identifier |
| `image_name` | TEXT | Image identifier |
| `keypoint_x` | REAL | Associated keypoint x-coordinate |
| `keypoint_y` | REAL | Associated keypoint y-coordinate |
| `descriptor_vector` | BLOB | Binary cv::Mat descriptor data |
| `descriptor_dimension` | INTEGER | Descriptor length (e.g., 128 for SIFT) |
| `processing_method` | TEXT | Processing pipeline identifier |
| `normalization_applied` | TEXT | Normalization type: "NoNorm", "L2", "L1" |
| `rooting_applied` | TEXT | Rooting type: "NoRoot", "RBef", "RAft" |
| `pooling_applied` | TEXT | Pooling type: "None", "Dom", "Stack" |
| `created_at` | TIMESTAMP | Descriptor computation time |

## Database Indexes

Performance-optimized indexes for common query patterns:

```sql
-- Keypoint set queries
CREATE INDEX idx_keypoint_sets_method ON keypoint_sets(generation_method);
CREATE INDEX idx_locked_keypoints_set ON locked_keypoints(keypoint_set_id);
CREATE INDEX idx_locked_keypoints_scene ON locked_keypoints(keypoint_set_id, scene_name, image_name);

-- Descriptor queries  
CREATE INDEX idx_descriptors_experiment ON descriptors(experiment_id, processing_method);
CREATE INDEX idx_descriptors_keypoint ON descriptors(scene_name, image_name, keypoint_x, keypoint_y);
CREATE INDEX idx_descriptors_method ON descriptors(processing_method, normalization_applied, rooting_applied);
```

## Metric Definitions

### True IR-style mAP Metrics (Primary)

These metrics follow standard Information Retrieval evaluation practices:

- **Homography-based ground truth**: Uses HPatches homography matrices for correspondence
- **3-pixel tolerance**: Matches within 3px are considered correct (τ=3.0)
- **R=1 policy**: Single ground truth per query (binary relevance)
- **Standard AP formula**: `AP = (1/R) * Σ(Precision@k * 1[rel[k]=1])`

#### true_map_macro
- **Scene-balanced mAP**: Average AP across scenes (macro-averaging)
- **Primary evaluation metric** - Recommended for reporting
- **Handles scene imbalance**: Each scene contributes equally regardless of size

#### true_map_micro  
- **Overall mAP**: Weighted by total query count (micro-averaging)
- **Query-weighted evaluation**: Larger scenes have more influence
- **Higher values expected**: Typically ~0.2% higher than macro

#### true_map_macro_with_zeros
- **Conservative macro mAP**: Includes R=0 queries as AP=0
- **Punitive evaluation**: Penalizes poor descriptor quality
- **Research insight**: Shows impact of degenerate descriptors

#### true_map_micro_with_zeros
- **Conservative micro mAP**: Includes R=0 queries as AP=0
- **Query-weighted punitive**: Weighted by total queries including R=0

### Legacy Metrics (Compatibility)

#### legacy_mean_precision
- **Original arithmetic mean**: Simple average of per-image precisions
- **Backward compatibility**: Preserves v1.0 calculation method
- **Note**: Does not follow IR standards but useful for historical comparison

#### mean_average_precision
- **Primary display metric**: Uses `true_map_macro` when available, falls back to legacy
- **Transition support**: Allows gradual migration to new metrics
- **Recommended usage**: Use for backward-compatible reporting

## Query Examples

### Current Performance Analysis
```sql
-- Top performing experiments by true mAP
SELECT e.descriptor_type, e.pooling_strategy,
       r.true_map_macro, r.true_map_micro,
       r.processing_time_ms
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE r.true_map_macro IS NOT NULL
ORDER BY r.true_map_macro DESC
LIMIT 10;
```

### Pooling Strategy Comparison
```sql
-- Compare pooling strategies using macro mAP
SELECT e.pooling_strategy,
       AVG(r.true_map_macro) as avg_macro_map,
       AVG(r.true_map_micro) as avg_micro_map,
       COUNT(*) as experiment_count
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE r.true_map_macro IS NOT NULL
GROUP BY e.pooling_strategy
ORDER BY avg_macro_map DESC;
```

### Historical Compatibility
```sql
-- View both new and legacy metrics for comparison
SELECT e.descriptor_type,
       r.true_map_macro,
       r.legacy_mean_precision,
       (r.true_map_macro - r.legacy_mean_precision) as difference
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE r.true_map_macro IS NOT NULL 
  AND r.legacy_mean_precision IS NOT NULL;
```

### Keypoint Set Usage
```sql
-- Analyze keypoint sets by generation method
SELECT ks.name, ks.generation_method,
       COUNT(lk.id) as keypoint_count,
       COUNT(DISTINCT lk.scene_name) as scene_count
FROM keypoint_sets ks
LEFT JOIN locked_keypoints lk ON ks.id = lk.keypoint_set_id
GROUP BY ks.id, ks.name, ks.generation_method
ORDER BY keypoint_count DESC;
```

## API Usage

### DatabaseManager Methods (C++)

```cpp
// Record experiment with new metrics
ExperimentResults results;
results.true_map_macro = 0.3703;
results.true_map_micro = 0.3724;
results.true_map_macro_with_zeros = 0.3502;
results.true_map_micro_with_zeros = 0.3386;
results.legacy_mean_precision = 0.3607;  // backward compatibility
results.mean_average_precision = results.true_map_macro;  // primary display

db.recordExperiment(results);

// Query recent results with new metrics
auto recent = db.getRecentResults(10);
for (const auto& result : recent) {
    std::cout << "Macro mAP: " << result.true_map_macro << std::endl;
    std::cout << "Micro mAP: " << result.true_map_micro << std::endl;
}
```

### Analysis Scripts (Python)

```python
import sqlite3
import pandas as pd

# Load data with new metrics
query = """
SELECT e.descriptor_type, e.pooling_strategy,
       r.true_map_macro, r.true_map_micro,
       r.true_map_macro_with_zeros, r.true_map_micro_with_zeros,
       r.legacy_mean_precision
FROM experiments e
JOIN results r ON e.id = r.experiment_id
"""

df = pd.read_sql_query(query, sqlite3.connect("experiments.db"))

# Primary analysis using macro mAP
best_config = df.loc[df['true_map_macro'].idxmax()]
print(f"Best configuration: {best_config['descriptor_type']}")
print(f"Macro mAP: {best_config['true_map_macro']:.4f}")
```

## Migration Checklist

When upgrading from v1.0 to v2.0:

- [ ] **Backup database**: Migration script creates automatic backup
- [ ] **Run migration**: `python3 migrate_database.py <db_path>`
- [ ] **Verify columns**: Check that new columns exist and are populated
- [ ] **Update queries**: Modify analysis scripts to use primary columns
- [ ] **Test compatibility**: Ensure legacy applications still work
- [ ] **Update documentation**: Reflect new metrics in research papers

## Performance Considerations

- **Storage**: New columns add ~32 bytes per result row
- **Indexes**: No additional indexes needed for new columns
- **Query speed**: Primary column access is faster than metadata parsing
- **Migration time**: ~1 second per 1000 result rows

## Best Practices

1. **Primary metric**: Use `true_map_macro` for reporting performance
2. **Scene analysis**: Use `true_map_micro` for query-weighted insights
3. **Conservative evaluation**: Include "with_zeros" metrics in comprehensive analysis
4. **Backward compatibility**: Keep `legacy_mean_precision` for historical comparison
5. **Documentation**: Always specify which MAP variant is being reported

This schema provides a solid foundation for descriptor research with proper IR-style evaluation metrics while maintaining full backward compatibility.