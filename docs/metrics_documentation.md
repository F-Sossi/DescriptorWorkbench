# DescriptorWorkbench Metrics System Documentation

## Overview

The DescriptorWorkbench metrics system provides comprehensive evaluation metrics for image descriptor performance, implementing both traditional computer vision metrics and modern Information Retrieval (IR) style evaluation methods. The system is designed to support rigorous academic research and benchmarking of descriptor algorithms.

---

## **Core Metrics Architecture**

### **Key Files:**
- **`src/core/metrics/ExperimentMetrics.hpp`** - Main metrics container and computation logic
- **`src/core/metrics/TrueAveragePrecision.hpp/.cpp`** - IR-style mAP implementation
- **`src/core/metrics/KeypointVerification.hpp/.cpp`** - Bojanic verification/retrieval tasks
- **`src/core/metrics/MetricsCalculator.hpp`** - Utility functions for metrics aggregation

---

## **Metric Categories**

### **1. Legacy Precision Metrics**

#### **Mean Precision** (`mean_precision`)
- **Definition**: Simple arithmetic mean of per-image precision values
- **Formula**: `Σ(precision_i) / N_images`
- **Use Case**: Basic descriptor performance overview
- **Range**: [0.0, 1.0]
- **Example**: For precisions [0.8, 0.6, 0.9] → Mean = 0.767

#### **Legacy Macro Precision by Scene** (`legacy_macro_precision_by_scene`) 
- **Definition**: Macro average of per-scene mean precisions (scene-balanced)
- **Formula**: `Σ(scene_mean_precision) / N_scenes`
- **Use Case**: Balanced evaluation across different scene types  
- **Range**: [0.0, 1.0]
- **Note**: **NOT** Information Retrieval style mAP!

---

### **2. True Information Retrieval Style mAP**

#### **True Micro mAP** (`true_map_micro`)
- **Definition**: Standard IR-style Mean Average Precision over all queries
- **Formula**: `Σ(AP_query) / N_queries_processed`
- **Ground Truth**: Homography-based geometric correspondence (τ=3px tolerance)
- **Policy**: Single-GT (R=1) - each query has at most one relevant result
- **Excludes**: R=0 queries (no ground truth correspondence)
- **Use Case**: Overall retrieval effectiveness comparison
- **Range**: [0.0, 1.0]

#### **True Macro mAP by Scene** (`true_map_macro_by_scene`)
- **Definition**: Scene-balanced mAP (macro average across scenes)
- **Formula**: `Σ(scene_mAP) / N_scenes`
- **Use Case**: Prevents scene imbalance from dominating results
- **Range**: [0.0, 1.0]
- **Advantage**: Equal weight to all scene types regardless of query count

#### **Viewpoint/Illumination Split**
- **Viewpoint mAP** (`viewpoint_map`)
  - mAP for `v_*` sequences only (geometric changes)
  - Typically **lower** due to perspective distortion

- **Illumination mAP** (`illumination_map`)
  - mAP for `i_*` sequences only (photometric changes)
  - Typically **higher** (geometry unchanged)

#### **Punitive mAP Variants** (Including R=0 Queries)
- **Micro Including Zeros** (`true_map_micro_including_zeros`)
  - Includes R=0 queries as AP=0.0 in the average
  - Formula: `Σ(AP_all_queries) / N_total_queries`
  - **Lower** values than regular mAP (more conservative)

- **Macro Including Zeros** (`true_map_macro_by_scene_including_zeros`)
  - Scene-balanced version including R=0 penalties per scene
  - Penalizes descriptors that fail on many queries

---

### **3. Bojanic et al. (2020) Metrics**

These metrics implement the evaluation protocol from "Image Feature Matching: What Works" (Bojanic et al., 2020).

#### **Keypoint Verification** (`keypoint_verification_ap`)
- **Definition**: Tests discriminative power using distractor keypoints
- **Formula**: AP computed with distractors from other images
- **Columns**: `keypoint_verification_ap`, `verification_viewpoint_ap`, `verification_illumination_ap`
- **Use Case**: Tests if descriptor can distinguish true matches from distractors

#### **Keypoint Retrieval** (`keypoint_retrieval_ap`)
- **Definition**: Three-tier labeling system (y ∈ {-1, 0, +1})
  - `y = +1`: True positive (correct match)
  - `y = 0`: Hard negative (same image, different keypoint)
  - `y = -1`: Distractor (different image)
- **Columns**: `keypoint_retrieval_ap`, `retrieval_viewpoint_ap`, `retrieval_illumination_ap`
- **Additional Counts**: `retrieval_num_true_positives`, `retrieval_num_hard_negatives`, `retrieval_num_distractors`

---

### **4. Precision@K and Recall@K Metrics**

#### **Precision@K** (`precision_at_1`, `precision_at_5`, `precision_at_10`)
- **Definition**: Percentage of queries where true match is in top-K results
- **Formula**: `N_hits_at_K / N_valid_queries`
- **Excludes**: R=0 queries (no potential true match)
- **Use Case**: Practical retrieval performance at different cutoffs
- **Range**: [0.0, 1.0]

#### **Recall@K** (`recall_at_1`, `recall_at_5`, `recall_at_10`)
- **Definition**: For R=1 policy, Recall@K = Precision@K
- **Reason**: Single ground truth means precision and recall are equivalent
- **Use Case**: Consistency with multi-GT scenarios in other systems

**Example Interpretation:**
```
P@1 = 0.274 → 27.4% of queries have correct match as top result  
P@5 = 0.458 → 45.8% of queries have correct match in top-5 results
P@10 = 0.495 → 49.5% of queries have correct match in top-10 results
```

---

## **Metric Computation Process**

### **1. Ground Truth Establishment**
```cpp
// Homography-based correspondence
Point2D projected = projectPoint(H_A_to_B, queryA);
int gt_idx = findClosestKeypoint(projected, keypointsB, tau_px=3.0);
```

### **2. Average Precision Calculation**
```cpp
// For R=1 case (single ground truth)
double ap = 1.0 / rank_of_true_match;  // Optimized formula
```

### **3. Aggregation Methods**
- **Micro**: Average AP over all valid queries
- **Macro**: Average AP per scene, then average across scenes
- **Including Zeros**: Add R=0 queries as AP=0.0

---

## **Scene-Based Analysis**

### **Scene Types in HPatches Dataset**
- **`i_*` scenes**: Illumination changes (same viewpoint)
  - Examples: `i_dome`, `i_contruction`, `i_autannes`
  - **Typically higher** descriptor performance
- **`v_*` scenes**: Viewpoint changes (same illumination)  
  - Examples: `v_wall`, `v_churchill`, `v_beyus`
  - **Typically lower** descriptor performance due to geometric changes

### **Per-Scene Metrics Available**
```cpp
// Individual scene analysis
double dome_precision = metrics.getSceneAveragePrecision("i_dome");
std::vector<std::string> scenes = metrics.getSceneNames();

// Scene-specific counts
int dome_matches = metrics.per_scene_matches["i_dome"];
int dome_keypoints = metrics.per_scene_keypoints["i_dome"];
int dome_images = metrics.per_scene_image_count["i_dome"];
```

---

##  **Real-World Example Analysis**

Based on actual experiment results on HPatches:

### **SIFT-Family Performance (scale-matched keypoints):**
```
Descriptor          mAP     Viewpoint   Illumination
-------------------------------------------------
SIFT (baseline)    63.86%    65.66%      59.79%
DSPSIFT_V2         65.31%    66.75%      60.59%
DSPRGBSIFT_V2      66.03%    67.55%      61.90%
RGBSIFT            64.69%    66.50%      61.14%
```

### **With Intersection Filtering:**
```
Descriptor          mAP     Keypoint Set
-------------------------------------------------
DSPSIFT_V2         74.93%   sift_surf_intersection
RGBSIFT            75.03%   sift_surf_intersection
SURF               75.08%   sift_surf_intersection
```

### **Performance Interpretation:**
- **Viewpoint > Illumination**: Geometric changes typically harder than photometric
- **Scale filtering**: +21% absolute improvement (42.64% → 63.86%)
- **Intersection filtering**: Additional +10% on top of scale filtering

---

##  **Performance Optimization Features**

### **1. Efficient R=1 AP Computation**
- **Standard**: O(N log N) due to full sorting
- **Optimized**: O(N) rank computation for single ground truth
```cpp
// Count better matches (no sort needed)
int rank = 1 + count_if(distances, [gt_dist](double d) { return d < gt_dist; });
double ap = 1.0 / rank;  // Direct formula
```

### **2. Batch Processing Support**
```cpp
// Merge multiple experiment results
ExperimentMetrics overall;
for (const auto& folder_result : folder_results) {
    overall.merge(folder_result);
}
overall.calculateMeanPrecision();
```

### **3. Memory-Efficient Storage**
- Per-scene breakdown without data duplication
- Incremental AP accumulation
- Streaming-friendly rank calculation

---

## **Configuration Parameters**

### **Geometric Matching**
- **Pixel Tolerance**: `τ = 3.0px` (HPatches standard)
- **Homography**: Ground truth correspondence transformation
- **Boundary Filtering**: 40px border exclusion on keypoint generation

### **Evaluation Policy**
- **Single-GT (R=1)**: Each query has at most one relevant result  
- **Distance-based Ranking**: Descriptor L2 distance determines ranking
- **Scene-balanced**: Equal weight to all scene types

---

## **Usage Examples**

### **Basic Metrics Collection**
```cpp
ExperimentMetrics metrics;

// Add legacy precision results
metrics.addImageResult("i_dome", 0.75, 120, 200);
metrics.addImageResult("v_wall", 0.45, 80, 180);

// Add IR-style AP results  
TrueAveragePrecision::QueryAPResult ap_result = 
    TrueAveragePrecision::computeQueryAP(queryKP, homography, targetKPs, distances);
metrics.addQueryAP("i_dome", ap_result);

// Calculate final metrics
metrics.calculateMeanPrecision();
```

### **Retrieving Results**
```cpp
// Main performance indicators
double micro_map = metrics.true_map_micro;        // Overall effectiveness
double macro_map = metrics.true_map_macro_by_scene; // Scene-balanced
double p_at_1 = metrics.precision_at_1;           // Exact match rate
double p_at_5 = metrics.precision_at_5;           // Top-5 success rate

// Diagnostic information
int valid_queries = metrics.total_queries_processed;
int excluded_queries = metrics.total_queries_excluded; // R=0 cases
double processing_time = metrics.processing_time_ms;
```

### **Scene-Level Analysis**
```cpp
// Compare illumination vs viewpoint performance
double illum_perf = 0.0, view_perf = 0.0;
int illum_count = 0, view_count = 0;

for (const auto& scene : metrics.getSceneNames()) {
    double scene_ap = metrics.getSceneAveragePrecision(scene);
    if (scene.substr(0, 2) == "i_") {
        illum_perf += scene_ap; illum_count++;
    } else if (scene.substr(0, 2) == "v_") {
        view_perf += scene_ap; view_count++;
    }
}

illum_perf /= illum_count;  // Average illumination scene performance
view_perf /= view_count;    // Average viewpoint scene performance
```

---

## 📊 **Metric Comparison Guidelines**

### **Which Metric to Use When:**

| **Use Case** | **Recommended Metric** | **Reason** |
|--------------|------------------------|------------|
| **Overall Algorithm Comparison** | `true_map_micro` | Standard IR evaluation, comparable to literature |
| **Scene-Balanced Evaluation** | `true_map_macro_by_scene` | Prevents scene count imbalance |
| **Conservative/Punitive Eval** | `*_including_zeros` | Penalizes descriptor failures |
| **Practical Application** | `precision_at_5` | Real-world retrieval cutoff |
| **Legacy Comparison** | `mean_precision` | Backward compatibility |

### **Typical Value Ranges (with scale-filtered keypoints):**
- **Excellent Descriptor**: mAP > 70%, Best intersection sets achieve 75%+
- **Good Descriptor**: mAP 60-70%, Most DSP variants
- **Baseline**: mAP ~64%, Standard SIFT on scale-matched keypoints
- **Poor (unfiltered)**: mAP ~43%, SIFT on all keypoints (no scale filtering)

---

## **Advanced Features**

### **Error Handling and Robustness**
```cpp
// Create error metrics for failed experiments
ExperimentMetrics error_result = ExperimentMetrics::createError("Feature detection failed");

// Merge handles failures gracefully
metrics.merge(other_metrics);  // Combines success/failure states
```

### **Debugging and Diagnostics**
- **Rank Histograms**: Distribution of true match rankings
- **R=0 Query Tracking**: Count of queries without ground truth
- **Per-Scene Breakdown**: Detailed performance by scene type
- **Processing Timing**: Performance profiling support

### **Metadata Export**
All metrics are serialized to database metadata field for analysis:
```
true_map_micro=0.3724;true_map_macro_by_scene=0.3703;precision_at_1=0.2833;
total_queries_processed=215939;total_queries_excluded=21536;
i_dome_true_map=0.3598;v_wall_true_map=0.4579;...
```

---

## **Academic Context**

### **Compliance with Standards**
- **HPatches Methodology**: Uses standard τ=3px tolerance and homography ground truth
- **IR-Style mAP**: Follows Information Retrieval evaluation conventions
- **Reproducible**: Deterministic ground truth generation with locked keypoints

### **Citation-Ready Results**
The metrics system supports rigorous academic evaluation with:
- Statistical significance testing capability
- Scene-balanced evaluation
- Multiple complementary metrics
- Detailed diagnostic information

### **Benchmarking Support**
- Cross-algorithm comparison through standardized metrics
- Scene-type analysis for understanding algorithm strengths/weaknesses  
- Processing time tracking for efficiency evaluation
- Extensible for new descriptor types and pooling strategies

---

## **Integration Points**

### **Database Storage**
```sql
-- Main metrics stored in results table (schema v3.3)
CREATE TABLE results (
    -- Primary IR-style mAP
    true_map_micro REAL,
    true_map_macro REAL,

    -- Viewpoint/Illumination split
    viewpoint_map REAL,
    illumination_map REAL,

    -- Bojanic metrics
    keypoint_verification_ap REAL,
    keypoint_retrieval_ap REAL,

    -- Precision@K
    precision_at_1 REAL,
    precision_at_5 REAL,

    -- Legacy
    mean_average_precision REAL,

    -- Timing
    processing_time_ms REAL,
    metadata TEXT
);
```

### **Analysis Integration**
- **Jupyter Notebooks**: Direct pandas DataFrame integration
- **CLI Tools**: Real-time metrics display during experiments  
- **Database Queries**: SQL-accessible for custom analysis
- **Export Formats**: CSV/JSON export for external analysis tools

This comprehensive metrics system enables both practical descriptor evaluation and rigorous academic research, providing multiple perspectives on descriptor performance while maintaining compatibility with established benchmarking standards.