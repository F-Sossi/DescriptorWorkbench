# YAML Configuration → Code Implementation Mapping

This document provides a complete mapping of YAML configuration settings to their implementation in the experiment runner pipeline.

---

## **1. EXPERIMENT Section**

### YAML Settings:
```yaml
experiment:
  name: "sift_baseline"
  description: "Baseline SIFT descriptor"
  version: "1.0"
  author: "automation"
```

### Code Flow:
- **Parsing**: `YAMLConfigLoader.cpp:65-70` → `parseExperiment()`
- **Storage**: `ExperimentConfig.hpp:18-23` → `Experiment` struct
- **Usage**: `experiment_runner.cpp:483-484` → Logged to console
- **Database**: `experiment_runner.cpp:503` → Stored in `parameters["experiment_name"]`

---

## **2. DATASET Section**

### YAML Settings:
```yaml
dataset:
  type: "hpatches"
  path: "../data/"
  scenes: []  # Empty = all scenes, or ["i_dome", "v_wall"] for specific
```

### Code Flow:
- **Parsing**: `YAMLConfigLoader.cpp:72-82` → `parseDataset()`
- **Storage**: `ExperimentConfig.hpp:26-30` → `Dataset` struct
- **Validation**: `experiment_runner.cpp:140-142` → Check path exists
- **Scene Filtering**: `experiment_runner.cpp:317-323` → Skip scenes not in list if specified
- **Database**: `experiment_runner.cpp:499` → Stored in `dataset_path`

---

## **3. KEYPOINTS Section**

### YAML Settings:
```yaml
keypoints:
  generator: "sift"              # or "harris", "orb", "keynet", "locked_in"
  source: "locked_in"            # or "homography_projection", "independent_detection"
  keypoint_set_name: "sift_keynet_pairs"
  use_locked_keypoints: true     # Legacy, prefer 'source'

keypoints_params:
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
  sigma: 1.6
  num_octaves: 4
```

### Code Flow:

#### **generator** (KeypointGenerator enum):
- **Parsing**: `YAMLConfigLoader.cpp:85-87` → `stringToKeypointGenerator()` at line 407-414
- **Storage**: `types.hpp:315-326` → `KeypointParams` struct
- **Usage**: `experiment_runner.cpp:121-126` → `makeDetector()` creates SIFT detector
- **Options**: SIFT, HARRIS, ORB, KEYNET, LOCKED_IN

#### **source** (KeypointSource enum):
- **Parsing**: `YAMLConfigLoader.cpp:108-110` → `keypointSourceFromString()` at `types.hpp:276-280`
- **Storage**: `types.hpp:322` → `KeypointParams.source`
- **Critical Usage Points**:
  1. `experiment_runner.cpp:145-147` → Determines if database keypoints are loaded
  2. `experiment_runner.cpp:383` → Controls if match correctness is evaluated
  3. `experiment_runner.cpp:336-342` vs `experiment_runner.cpp:363-369` → Load from DB vs detect fresh
- **Options**:
  - `HOMOGRAPHY_PROJECTION`: Use transformed keypoints from database (controlled evaluation)
  - `INDEPENDENT_DETECTION`: Detect fresh keypoints on each image (realistic evaluation)

#### **keypoint_set_name**:
- **Parsing**: `YAMLConfigLoader.cpp:111-113`
- **Resolution**: `experiment_runner.cpp:473-481` → Converts name to `keypoint_set_id` via database lookup
- **Usage**: `experiment_runner.cpp:232-246` → `loadKeypointsFromDatabase()` fetches keypoints by set ID

#### **max_features**:
- **Parsing**: `YAMLConfigLoader.cpp:90-92`
- **Storage**: `types.hpp:316` → `KeypointParams.max_features`
- **Usage**:
  - `experiment_runner.cpp:123-125` → Passed to `cv::SIFT::create(maxf)`
  - `experiment_runner.cpp:502` → Stored in database config

---

## **4. DESCRIPTORS Section** (Array of Descriptor Configs)

### YAML Settings:
```yaml
descriptors:
  - name: "sift_baseline"
    type: "sift"                          # See full list below
    pooling: "none"                       # or "dsp", "stacking"
    pooling_aggregation: "max"            # or "average", "min", "concatenate", "weighted_avg"
    rooting_stage: "after_pooling"        # or "before_pooling", "none"
    normalize_after_pooling: true
    normalize_before_pooling: false
    norm_type: 4                          # L2 norm (cv::NORM_L2)
    use_color: false

    # DSP-specific settings:
    scales: [1.0, 1.5, 2.0]
    scale_weights: [0.5, 0.3, 0.2]        # Optional weighted pooling
    scale_weighting: "gaussian"           # or "uniform", "triangular"
    scale_weight_sigma: 0.15

    # Stacking-specific:
    secondary_descriptor: "honc"
    stacking_weight: 0.5

    # DNN-specific:
    dnn:
      model: "path/to/model.onnx"
      input_size: 32
      support_multiplier: 1.0
      rotate_to_upright: true
      mean: 0.0
      std: 1.0
      per_patch_standardize: false
```

### Code Flow:

#### **Descriptor Type** (type):
- **Parsing**: `YAMLConfigLoader.cpp:134-136` → `stringToDescriptorType()` at lines 362-380
- **Supported Types** (from `types.hpp:61-79`):
  - `sift`, `rgbsift`, `vsift`, `honc`, `dspsift`, `dspsift_v2`, `dsprgbsift_v2`, `dsphowh_v2`, `dsphonc_v2`
  - `vgg`, `dnn_patch`, `libtorch_hardnet`, `libtorch_sosnet`, `libtorch_l2net`
  - `orb`, `surf`
- **Factory Creation**:
  - `experiment_runner.cpp:151-179` → Creates extractor
  - DNN descriptors: Direct construction with parameters (lines 151-176)
  - Others: `DescriptorFactory::create()` (line 178)

#### **Pooling Strategy** (pooling):
- **Parsing**: `YAMLConfigLoader.cpp:139-141` → `stringToPoolingStrategy()` at lines 382-387
- **Storage**: `types.hpp:329` → `DescriptorParams.pooling`
- **Usage**: `experiment_runner.cpp:180` → `PoolingFactory::createFromConfig(desc_config)`
- **Applied**: `experiment_runner.cpp:253-254` → `computeDescriptorsWithPooling()`
- **Options**: `none`, `domain_size_pooling` (or `dsp`), `stacking`

#### **Pooling Aggregation** (pooling_aggregation):
- **Parsing**: `YAMLConfigLoader.cpp:143-146` → `stringToPoolingAggregation()` at lines 389-397
- **Storage**: `types.hpp:330` → `DescriptorParams.pooling_aggregation`
- **Options**: `average`, `max`, `min`, `concatenate`, `weighted_avg`
- **Usage**: Controls how multi-scale descriptors are combined in DSP wrappers

#### **Scales** (scales):
- **Parsing**: `YAMLConfigLoader.cpp:148-153` → Parse float sequence
- **Storage**: `types.hpp:331` → `DescriptorParams.scales` (vector)
- **Validation**: `YAMLConfigLoader.cpp:258-265` → Must all be > 0.0
- **Usage**: Passed to pooling strategy for multi-scale processing

#### **Scale Weights** (scale_weights):
- **Parsing**: `YAMLConfigLoader.cpp:154-159`
- **Validation**: `YAMLConfigLoader.cpp:266-270` → Must match length of `scales`
- **Priority**: Takes precedence over `scale_weighting` (warning at line 279-281)

#### **Normalization** (normalize_before_pooling, normalize_after_pooling):
- **Parsing**: `YAMLConfigLoader.cpp:170-176`
- **Storage**: `types.hpp:337-338`
- **Usage**: Applied in pooling strategy and descriptor computation

#### **Rooting Stage** (rooting_stage) - RootSIFT:
- **Parsing**: `YAMLConfigLoader.cpp:178-181` → `stringToRootingStage()` at lines 399-405
- **Storage**: `types.hpp:339` → `DescriptorParams.rooting_stage`
- **Options**: `before_pooling`, `after_pooling`, `none`
- **Usage**: Applied during descriptor computation for RootSIFT transformation

#### **Norm Type** (norm_type):
- **Parsing**: `YAMLConfigLoader.cpp:191-196` → String to OpenCV constant
- **Storage**: `types.hpp:340`
- **Database**: `experiment_runner.cpp:506` → Stored as string
- **Options**: `"l1"` → `cv::NORM_L1`, `"l2"` → `cv::NORM_L2`, or integer (4 = L2)

#### **Color Usage** (use_color):
- **Parsing**: `YAMLConfigLoader.cpp:183-185`
- **Storage**: `types.hpp:342`
- **Usage**: `experiment_runner.cpp:331-333` and `358-360` → Converts to grayscale if `false`

#### **DNN Parameters** (dnn):
- **Parsing**: `YAMLConfigLoader.cpp:206-216` → Nested object
- **Storage**: `types.hpp:349-357` → Multiple DNN-specific params
- **Usage**: `experiment_runner.cpp:152-165` → Passed to `DNNPatchWrapper` constructor
- **Fallback**: `experiment_runner.cpp:167-175` → `PseudoDNNWrapper` if ONNX fails

---

## **5. EVALUATION Section**

### YAML Settings:
```yaml
evaluation:
  matching:
    method: "brute_force"          # or "flann", "ratio_test"
    norm: "l2"                     # or "l1"
    cross_check: true
    threshold: 0.8                 # Distance threshold or ratio threshold
    ratio_threshold: 0.8           # Alias for threshold when using ratio_test

  validation:
    method: "homography"           # or "cross_image", "none"
    threshold: 0.05                # Pixel threshold for homography validation
    min_matches: 10

  image_retrieval:
    enabled: true                  # Optional dataset-level retrieval MAP
    scorer: "total_matches"        # "total_matches" (default), "ratio_sum", or "correct_matches"
```

### Code Flow:

#### **Matching Method** (method):
- **Parsing**: `YAMLConfigLoader.cpp:306-308` → `stringToMatchingMethod()` at lines 416-421
- **Storage**: `types.hpp:360` → `EvaluationParams.matching_method`
- **Factory**: `experiment_runner.cpp:181-182` → `MatchingFactory::createStrategy()`
- **Usage**: `experiment_runner.cpp:384-385` → `computeMatches()` with selected strategy
- **Options**: `brute_force`, `flann`, `ratio_test` (SNN ratio test)

#### **Match Threshold** (threshold / ratio_threshold):
- **Parsing**: `YAMLConfigLoader.cpp:321-328` → Both keys supported
- **Storage**: `types.hpp:363` → `EvaluationParams.match_threshold`
- **Validation**: `YAMLConfigLoader.cpp:296-298` → Must be in [0, 1]
- **Database**: `experiment_runner.cpp:501` → Stored as `similarity_threshold`
- **Usage**: Passed to matching strategy for filtering

#### **Cross Check** (cross_check):
- **Parsing**: `YAMLConfigLoader.cpp:317-319`
- **Storage**: `types.hpp:362`
- **Usage**: Applied in matching strategy for bidirectional validation

#### **Validation Method** (validation.method):
- **Parsing**: `YAMLConfigLoader.cpp:335-337` → `stringToValidationMethod()` at lines 423-428
- **Storage**: `types.hpp:365`
- **Options**: `homography`, `cross_image`, `none`

#### **Validation Threshold** (validation.threshold):
- **Parsing**: `YAMLConfigLoader.cpp:339-341`
- **Storage**: `types.hpp:366` → Pixel threshold for geometric validation
- **Usage**: `experiment_runner.cpp:398-406` → `maybeAccumulateTrueAveragePrecisionFromFile()`

#### **Image Retrieval Metric** (image_retrieval.*):
- **Parsing**: `YAMLConfigLoader.cpp:369-378`
- **Storage**: `types.hpp:369-374` → `ImageRetrievalParams`
- **Default Scorer**: `"total_matches"` in `types.hpp`
- **Runtime Toggle**: `experiment_runner.cpp:648-759` instantiates `ImageRetrievalAccumulator`
- **Scorers**:
  - `total_matches` → raw match count (`cli/experiment_runner.cpp:314-330`, window line ~320)
  - `ratio_sum` → ∑ 1/(1 + distance)
  - `correct_matches` → requires `homography_projection` keypoints; otherwise 0
  - Unknown strings fall back to `total_matches`
- **Ranking Logic**: accumulator recomputes query vs candidate matches, sorts by score, builds relevance list via scene match, computes AP (`cli/experiment_runner.cpp:204-348`)
- **Aggregation**: per-query retrieval AP stored in `ExperimentMetrics::addImageRetrievalAP` (`src/core/metrics/ExperimentMetrics.hpp:295`); final MAP computed in `calculateMeanPrecision` (`src/core/metrics/ExperimentMetrics.hpp:241-248`)
- **Database**: stored as `results.image_retrieval_map` (`database/schema.sql:33-42`, `src/core/database/DatabaseManager.cpp:445-486`)
- **Performance Note**: Enables full cross-scene matching; OpenMP disabled for this mode (`cli/experiment_runner.cpp:705-719`)

---

## **6. DATABASE Section**

### YAML Settings:
```yaml
database:
  connection: "sqlite:///experiments.db"
  save_descriptors: false
  save_matches: false
  save_visualizations: false
```

### Code Flow:

#### **Connection String** (connection):
- **Parsing**: `YAMLConfigLoader.cpp:352`
- **Normalization**: `experiment_runner.cpp:448-458` → Strips `sqlite:///` prefix
- **Usage**: `experiment_runner.cpp:460` → `DatabaseManager` initialization
- **Default**: `"experiments.db"` if empty

#### **Save Descriptors** (save_descriptors):
- **Parsing**: `YAMLConfigLoader.cpp:354`
- **Storage**: `types.hpp:373`
- **Usage**: `experiment_runner.cpp:184-186` → Controls `store_descriptors` flag
- **Action**: `experiment_runner.cpp:264-273` → `maybeStoreDescriptors()` saves to DB

#### **Save Matches** (save_matches):
- **Parsing**: `YAMLConfigLoader.cpp:355`
- **Storage**: `types.hpp:374`
- **Usage**: `experiment_runner.cpp:186-187` → Controls `store_matches` flag
- **Action**: `experiment_runner.cpp:275-285` → `maybeStoreMatches()` saves match pairs

#### **Save Visualizations** (save_visualizations):
- **Parsing**: `YAMLConfigLoader.cpp:356`
- **Storage**: `types.hpp:375`
- **Usage**: `experiment_runner.cpp:188-189` → Controls `store_visualizations` flag
- **Action**: `experiment_runner.cpp:287-309` → `maybeStoreVisualization()` generates and saves match images

---

## **Critical Implementation Details**

### **Keypoint Loading Logic** (Lines 336-369):
```cpp
if (use_db_keypoints) {  // Based on keypoints.source setting
    loadKeypointsFromDatabase(scene_name, image_name, keypoints);
} else {
    detectKeypoints(image, scene_name, image_name, keypoints);
}
```

### **Descriptor Computation Pipeline** (Lines 248-262):
```cpp
descriptors = computeDescriptorsWithPooling(
    image, keypoints, *extractor, *pooling, desc_config
);
```
- Respects: `use_color`, `normalize_before_pooling`, `normalize_after_pooling`, `rooting_stage`

### **Match Evaluation** (Lines 383-392):
```cpp
bool evaluateCorrectness = (source == HOMOGRAPHY_PROJECTION);
auto artifacts = computeMatches(descriptors1, descriptors2, *matcher, evaluateCorrectness);
```
- Only evaluates match correctness for `HOMOGRAPHY_PROJECTION` source

### **Results Storage** (Lines 526-606):
- Primary IR-style mAP metrics: `true_map_macro`, `true_map_micro` (lines 533-536)
- Legacy precision: `legacy_mean_precision` (line 544)
- Precision@K and Recall@K: stored in metadata (lines 571-576)
- Per-scene breakdown: stored with `_true_map` suffix (lines 583-600)

---

## **Validation Summary**

All YAML settings are properly:
1. ✅ **Parsed** by `YAMLConfigLoader`
2. ✅ **Validated** before execution (schema validation at line 222-299)
3. ✅ **Applied** in the experiment runner pipeline
4. ✅ **Stored** in the database for reproducibility

The configuration system is well-designed with clear separation of concerns and comprehensive error handling.

---

## **Quick Reference: All Supported Values**

### Descriptor Types:
`sift`, `rgbsift`, `vsift`, `honc`, `dspsift`, `dspsift_v2`, `dsprgbsift_v2`, `dsphowh_v2`, `dsphonc_v2`, `vgg`, `dnn_patch`, `libtorch_hardnet`, `libtorch_sosnet`, `libtorch_l2net`, `orb`, `surf`

### Pooling Strategies:
`none`, `domain_size_pooling` (or `dsp`), `stacking`

### Pooling Aggregation:
`average`, `max`, `min`, `concatenate`, `weighted_avg`

### Keypoint Generators:
`sift`, `harris`, `orb`, `keynet`, `locked_in`

### Keypoint Sources:
`homography_projection`, `independent_detection`

### Matching Methods:
`brute_force`, `flann`, `ratio_test`

### Validation Methods:
`homography`, `cross_image`, `none`

### Rooting Stages:
`before_pooling`, `after_pooling`, `none`

### Scale Weighting:
`uniform`, `triangular`, `gaussian`
