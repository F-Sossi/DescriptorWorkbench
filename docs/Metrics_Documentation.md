# DescriptorWorkbench Metrics System Documentation

## Overview

The DescriptorWorkbench metrics system provides comprehensive evaluation metrics for image descriptor performance, implementing both traditional computer vision metrics and modern Information Retrieval (IR) style evaluation methods. The system is designed to support rigorous academic research and benchmarking of descriptor algorithms.

---

## **Core Metrics Architecture**

### **Key Files:**
- **`src/core/metrics/ExperimentMetrics.hpp`** - Main metrics container and computation logic
- **`src/core/metrics/TrueAveragePrecision.hpp/.cpp`** - IR-style mAP implementation  
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

#### **Punitive mAP Variants** (Including R=0 Queries)
- **Micro Including Zeros** (`true_map_micro_including_zeros`)
  - Includes R=0 queries as AP=0.0 in the average
  - Formula: `Σ(AP_all_queries) / N_total_queries`
  - **Lower** values than regular mAP (more conservative)
  
- **Macro Including Zeros** (`true_map_macro_by_scene_including_zeros`)
  - Scene-balanced version including R=0 penalties per scene
  - Penalizes descriptors that fail on many queries

---

### **3. Precision@K and Recall@K Metrics**

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

Based on actual experiment results:

### **SIFT + DSP Pooling Performance:**
```
Legacy Mean Precision: 0.3703
True Micro mAP:       0.3724  
True Macro mAP:       0.3703
P@1:                  0.2833 (28.3% exact match rate)
P@5:                  0.4681 (46.8% top-5 match rate)
Processing Time:      202.0 seconds
```

### **Performance Interpretation:**
- **Micro > Macro mAP**: Some scenes perform better than others (unbalanced)
- **P@5 > P@1**: Many correct matches rank in positions 2-5 (descriptor ranking can improve)
- **~37% mAP**: Moderate performance, typical for SIFT on HPatches

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

### **Typical Value Ranges:**
- **Excellent Descriptor**: mAP > 0.5, P@5 > 0.7
- **Good Descriptor**: mAP 0.3-0.5, P@5 0.5-0.7  
- **Poor Descriptor**: mAP < 0.3, P@5 < 0.5

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
-- Main metrics stored in results table
CREATE TABLE results (
    mean_average_precision REAL,    -- Legacy macro precision
    precision_at_1 REAL,           -- P@1 
    precision_at_5 REAL,           -- P@5
    processing_time_ms REAL,       -- Timing
    metadata TEXT                  -- Full metrics serialization
);
```

### **Analysis Integration**
- **Jupyter Notebooks**: Direct pandas DataFrame integration
- **CLI Tools**: Real-time metrics display during experiments  
- **Database Queries**: SQL-accessible for custom analysis
- **Export Formats**: CSV/JSON export for external analysis tools

This comprehensive metrics system enables both practical descriptor evaluation and rigorous academic research, providing multiple perspectives on descriptor performance while maintaining compatibility with established benchmarking standards.