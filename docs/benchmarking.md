# Benchmarking & Metrics Reference

This document describes the evaluation metrics and timing data recorded during experiments.

---

## Core Metrics Files

| File | Purpose |
|------|---------|
| `src/core/metrics/ExperimentMetrics.hpp` | Main metrics container and aggregation logic |
| `src/core/metrics/TrueAveragePrecision.cpp/hpp` | IR-style mAP computation with homography ground truth |
| `src/core/metrics/KeypointVerification.cpp/hpp` | Bojanic et al. verification/retrieval tasks |
| `src/core/metrics/MetricsCalculator.hpp` | Utility functions for metrics aggregation |

---

## Primary Evaluation Metrics

### Image Matching (True mAP)

The primary benchmark is Information Retrieval-style Mean Average Precision using homography-projected ground truth.

| Metric | Database Column | Description |
|--------|-----------------|-------------|
| **True mAP (micro)** | `true_map_micro` | Overall mAP weighted by query count |
| **True mAP (macro)** | `true_map_macro` | Scene-balanced mAP (equal weight per scene) |
| Viewpoint mAP | `viewpoint_map` | mAP for `v_*` sequences (geometric changes) |
| Illumination mAP | `illumination_map` | mAP for `i_*` sequences (photometric changes) |

**Implementation**: `src/core/metrics/TrueAveragePrecision.cpp:87-131`

```cpp
// Ground truth: project query keypoint via homography, find nearest in target
const int gt_idx = findSingleRelevantIndex(queryA, H_A_to_B, keypointsB, tau_px);

// AP = 1/rank for single ground truth (R=1 policy)
int rank = 1 + count_better_matches;
result.ap = 1.0 / static_cast<double>(rank);
```

**Parameters**:
- Pixel tolerance: tau = 3.0px (HPatches standard)
- Policy: Single-GT (R=1) - each query has at most one correct match

### Precision@K

| Metric | Database Column | Description |
|--------|-----------------|-------------|
| P@1 | `precision_at_1` | % of queries where top result is correct |
| P@5 | `precision_at_5` | % of queries with correct match in top-5 |
| R@1 | `recall_at_1` | Same as P@1 for R=1 policy |
| R@5 | `recall_at_5` | Same as P@5 for R=1 policy |

---

## Bojanic et al. (2020) Metrics

These metrics implement the evaluation protocol from "Image Feature Matching: What Works" (Bojanic et al., 2020).

### Keypoint Verification

Tests discriminative power using distractor keypoints.

| Metric | Database Column | Description |
|--------|-----------------|-------------|
| Verification AP | `keypoint_verification_ap` | Overall verification AP |
| VP Verification | `verification_viewpoint_ap` | Verification for viewpoint scenes |
| IL Verification | `verification_illumination_ap` | Verification for illumination scenes |

**Implementation**: `src/core/metrics/KeypointVerification.cpp`

### Keypoint Retrieval

Three-tier labeling system (y in {-1, 0, +1}).

| Metric | Database Column | Description |
|--------|-----------------|-------------|
| Retrieval AP | `keypoint_retrieval_ap` | Overall retrieval AP |
| VP Retrieval | `retrieval_viewpoint_ap` | Retrieval for viewpoint scenes |
| IL Retrieval | `retrieval_illumination_ap` | Retrieval for illumination scenes |
| True Positives | `retrieval_num_true_positives` | Count of y=+1 labels |
| Hard Negatives | `retrieval_num_hard_negatives` | Count of y=0 labels |
| Distractors | `retrieval_num_distractors` | Count of y=-1 labels |

---

## Timing Metrics

| Metric | Database Column | Description |
|--------|-----------------|-------------|
| Total time | `processing_time_ms` | Total experiment runtime |
| Descriptor CPU | `descriptor_time_cpu_ms` | Descriptor extraction (CPU) |
| Descriptor GPU | `descriptor_time_gpu_ms` | Descriptor extraction (GPU) |
| Matching CPU | `match_time_cpu_ms` | Descriptor matching (CPU) |
| Matching GPU | `match_time_gpu_ms` | Descriptor matching (GPU) |
| Pipeline CPU | `total_pipeline_cpu_ms` | Full pipeline (CPU) |
| Pipeline GPU | `total_pipeline_gpu_ms` | Full pipeline (GPU) |

Additional timing data stored in `metadata` column as `key=value;` pairs.

---

## Database Schema

### experiments table

```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY,
    descriptor_type TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    pooling_strategy TEXT,
    keypoint_set_id INTEGER,       -- FK to keypoint_sets
    descriptor_dimension INTEGER,
    timestamp TEXT NOT NULL
);
```

### results table

```sql
CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER,         -- FK to experiments

    -- Primary metrics
    true_map_micro REAL,
    true_map_macro REAL,
    viewpoint_map REAL,
    illumination_map REAL,

    -- Bojanic metrics
    keypoint_verification_ap REAL,
    keypoint_retrieval_ap REAL,

    -- Precision@K
    precision_at_1 REAL,
    precision_at_5 REAL,

    -- Timing
    processing_time_ms REAL,

    -- Legacy (backward compatibility)
    mean_average_precision REAL,   -- Uses true_map_macro when available
    legacy_mean_precision REAL
);
```

---

## Querying Results

### Top experiments by mAP

```sql
SELECT
    e.descriptor_type,
    ks.name as keypoint_set,
    ROUND(r.true_map_micro * 100, 2) as mAP,
    ROUND(r.viewpoint_map * 100, 2) as VP,
    ROUND(r.illumination_map * 100, 2) as IL
FROM results r
JOIN experiments e ON r.experiment_id = e.id
LEFT JOIN keypoint_sets ks ON e.keypoint_set_id = ks.id
WHERE r.true_map_micro IS NOT NULL
ORDER BY r.true_map_micro DESC
LIMIT 20;
```

### Compare descriptors on same keypoint set

```sql
SELECT
    e.descriptor_type,
    ROUND(r.true_map_micro * 100, 2) as mAP,
    ROUND(r.precision_at_1 * 100, 2) as "P@1"
FROM results r
JOIN experiments e ON r.experiment_id = e.id
JOIN keypoint_sets ks ON e.keypoint_set_id = ks.id
WHERE ks.name = 'sift_scale_matched_6px'
ORDER BY r.true_map_micro DESC;
```

### Get Bojanic metrics

```sql
SELECT
    e.descriptor_type,
    ROUND(r.keypoint_verification_ap * 100, 2) as "Verif AP",
    ROUND(r.keypoint_retrieval_ap * 100, 2) as "Retr AP"
FROM results r
JOIN experiments e ON r.experiment_id = e.id
WHERE r.keypoint_verification_ap > 0
ORDER BY r.keypoint_verification_ap DESC;
```

---

## Reference Baselines

From experiments on HPatches dataset:

| Descriptor | Keypoint Set | mAP | Notes |
|------------|--------------|-----|-------|
| SIFT | sift_8000 (unfiltered) | 42.64% | Baseline |
| SIFT | sift_scale_matched_6px | 63.86% | +50% from scale filtering |
| DSPSIFT_V2 | sift_scale_matched_6px | 65.31% | +DSP pooling |
| DSPRGBSIFT_V2 | sift_scale_matched_6px | 66.03% | +Color |
| DSPSIFT_V2 | sift_surf_intersection | 74.93% | +Intersection filtering |
| RGBSIFT | sift_surf_intersection | 75.03% | Best SIFT-family |

**Key finding**: Scale filtering provides the largest improvement (+21% absolute).
