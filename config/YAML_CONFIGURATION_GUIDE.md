# DescriptorWorkbench YAML Configuration Guide

This guide provides comprehensive documentation for configuring experiments in the DescriptorWorkbench project using YAML files.

**🎉 Updated October 2025**: Complete documentation for Bojanic et al. (2020) evaluation metrics (keypoint verification + retrieval) and performance tuning options.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Schema Overview](#schema-overview)
3. [Configuration Sections](#configuration-sections)
4. [Evaluation Metrics (Bojanic et al. 2020)](#evaluation-metrics-bojanic-et-al-2020)
5. [Performance Tuning](#performance-tuning)
6. [Advanced Usage Patterns](#advanced-usage-patterns)
7. [Validation and Error Handling](#validation-and-error-handling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Minimal Configuration
```yaml
experiment: { name: "my_experiment" }
dataset: { type: "hpatches", path: "../data/" }
keypoints: { generator: "sift", source: "homography_projection" }
descriptors:
  - { name: "sift_baseline", type: "sift", pooling: "none", normalize_after_pooling: true }
evaluation: { matching: { method: "ratio_test", ratio_threshold: 0.8 } }
```

### Run an Experiment
```bash
cd build
./experiment_runner ../config/experiments/your_config.yaml
```

## Schema Overview

The DescriptorWorkbench uses **Schema v1** with these main sections:

| Section | Purpose | Required |
|---------|---------|----------|
| `experiment` | Metadata and identification | ✅ |
| `dataset` | Dataset configuration | ✅ |
| `keypoints` | Keypoint detection setup | ✅ |
| `descriptors` | Descriptor algorithms (array) | ✅ |
| `evaluation` | Matching, validation, and metrics | ✅ |
| `database` | Database storage settings | ❌ |
| `performance` | Parallelization and optimization | ❌ |

## Configuration Sections

### 1. Experiment Section

**Purpose**: Experiment metadata for tracking and documentation.

```yaml
experiment:
  name: "string"           # REQUIRED: Unique experiment identifier
  description: "string"    # Optional: Human-readable description
  version: "string"        # Optional: Version (default: "1.0")
  author: "string"         # Optional: Author name
```

**Best Practices**:
- Use descriptive names: `"sift_vs_cnn_comparison"` not `"test1"`
- Include key parameters in description: `"SIFT baseline with ratio test 0.8"`

### 2. Dataset Section

**Purpose**: Specify which dataset and scenes to evaluate.

```yaml
dataset:
  type: "hpatches"         # Only "hpatches" currently supported
  path: "../data/"         # REQUIRED: Path to HPatches dataset
  scenes: []               # Optional: Specific scenes ([] = all scenes)
```

**Scene Categories**:
- **Illumination changes**: `i_ajuntament`, `i_crownnight`, `i_dc`, `i_dome`, etc.
- **Viewpoint changes**: `v_boat`, `v_dogman`, `v_girl`, `v_wall`, etc.

**Examples**:
```yaml
# Evaluate all scenes (slow but comprehensive)
scenes: []

# Quick test on single scene
scenes: ["i_dome"]

# Compare illumination vs viewpoint robustness
scenes: ["i_dome", "v_wall"]

# Focus on illumination changes only
scenes: ["i_ajuntament", "i_crownnight", "i_dc", "i_dome"]
```

### 3. Keypoints Section

**Purpose**: Configure keypoint detection strategy and parameters.

```yaml
keypoints:
  generator: "sift"                    # Detector: "sift", "harris", "orb", "keynet"
  max_features: 2000                   # Max keypoints (0 = unlimited)
  source: "homography_projection"      # Strategy: see below
  keypoint_set_name: "sift_homo"      # Database identifier

  # SIFT-specific parameters
  contrast_threshold: 0.04             # Lower = more keypoints
  edge_threshold: 10.0                 # Higher = fewer edge keypoints
  sigma: 1.6                          # Gaussian scale
  num_octaves: 4                      # Scale space levels
```

**Keypoint Source Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `"homography_projection"` | Same keypoints transformed across images | Controlled evaluation of descriptor quality only |
| `"independent_detection"` | Fresh detection per image | Realistic evaluation including detector variability |

**Generator Options**:

| Generator | Description | Best For |
|-----------|-------------|----------|
| `"sift"` | OpenCV SIFT detector | General purpose, traditional descriptors |
| `"harris"` | Harris corner detector | Fast detection, corner-focused |
| `"orb"` | ORB detector | Speed-critical applications |
| `"keynet"` | CNN-based detector | CNN descriptors, research |

### 4. Descriptors Section

**Purpose**: Define one or more descriptor algorithms for comparison.

#### Basic Descriptor Configuration

```yaml
descriptors:
  - name: "unique_name"              # REQUIRED: Unique identifier
    type: "sift"                     # REQUIRED: Algorithm type
    pooling: "none"                  # Pooling strategy
    normalize_after_pooling: true    # L2 normalization
    device: "auto"                   # Processing device
```

#### Available Descriptor Types

| Type | Description | Pooling Support | Device |
|------|-------------|-----------------|---------|
| `"sift"` | OpenCV SIFT | ✅ | CPU |
| `"rgbsift"` | RGB color SIFT | ✅ | CPU |
| `"vsift"` | Vanilla SIFT implementation | ✅ | CPU |
| `"honc"` | Histogram of Normalized Colors | ✅ | CPU |
| `"dspsift"` | Domain-Size Pooled SIFT | ❌ | CPU |
| `"vgg"` | VGG descriptor | ✅ | CPU |
| `"libtorch_hardnet"` | HardNet CNN | ✅ | CPU/GPU |
| `"libtorch_sosnet"` | SOSNet CNN | ✅ | CPU/GPU |
| `"dnn_patch"` | ONNX-based CNN | ✅ | CPU |

**SURF extended (128D)**: add `extended: true` to a SURF descriptor (or composite component) to use 128-dimensional SURF instead of the 64D default. Useful for dimension-safe fusion (e.g., weighted_avg with SIFT/DSPSIFT at 128D).

#### Pooling Strategies

##### No Pooling (Default)
```yaml
- name: "sift_basic"
  type: "sift"
  pooling: "none"                    # Standard single-scale extraction
```

##### Domain Size Pooling
```yaml
# Explicit weights
- name: "sift_dsp_weighted"
  type: "sift"
  pooling: "domain_size_pooling"
  scales: [0.7, 1.0, 1.4]           # Scale factors
  scale_weights: [0.2, 0.6, 0.2]    # Must match scales length

# Gaussian weighting
- name: "sift_dsp_gaussian"
  type: "sift"
  pooling: "domain_size_pooling"
  scales: [0.5, 0.7, 1.0, 1.4, 2.0]
  scale_weighting: "gaussian"        # "uniform", "triangular", "gaussian"
  scale_weight_sigma: 0.15          # Gaussian width
```

##### Stacking Pooling
```yaml
- name: "stack_sift_rgb"
  type: "sift"                       # Primary descriptor
  pooling: "stacking"
  secondary_descriptor: "rgbsift"    # Secondary descriptor to stack
  stacking_weight: 0.5              # Combination weight [0,1]
```

#### Device Configuration

```yaml
# Auto-detect (recommended)
device: "auto"                       # Uses GPU if available, CPU fallback

# Force CPU (for fair comparisons)
device: "cpu"                        # Always use CPU

# Force GPU (for speed)
device: "cuda"                       # Requires CUDA-capable GPU
```

#### CNN Descriptor Advanced Options

```yaml
- name: "hardnet_advanced"
  type: "libtorch_hardnet"
  device: "cuda"
  dnn:
    input_size: 32                   # Patch size (32x32)
    support_multiplier: 1.0          # Patch scale multiplier
    rotate_to_upright: true          # Canonical orientation
    per_patch_standardize: true      # Z-score normalization

#### Composite Aggregation Dimensionality Rules

- `weighted_avg` / `average` / `max` / `min`: **All components must have the same descriptor dimension.** For SURF+SIFT/DSPSIFT fusion, set `extended: true` on SURF to reach 128D or use `concatenate` instead.
- `concatenate`: Safe with mixed dimensions; output dim is the sum of component dims.
- `channel_wise`: Special-case for 128D grayscale + 384D RGB (see composite color fusion examples).
```

### 5. Evaluation Section

**Purpose**: Configure matching and validation algorithms.

```yaml
evaluation:
  matching:
    method: "ratio_test"             # Matching algorithm
    ratio_threshold: 0.8             # Ratio test threshold [0,1]
    norm: "l2"                      # Distance norm: "l1", "l2"
    cross_check: false              # Mutual best matches

  validation:
    method: "homography"             # Validation method
    threshold: 0.05                  # Pixel error threshold
    min_matches: 10                 # Min matches for homography
```

#### Matching Methods

| Method | Description | Best For | Typical Threshold |
|--------|-------------|----------|-------------------|
| `"brute_force"` | Nearest neighbor | Simple baseline | 0.7-0.9 |
| `"ratio_test"` | Lowe's SNN ratio | Most descriptors | 0.8 |
| `"flann"` | Fast approximate | Large datasets | 0.7-0.9 |

#### Validation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `"homography"` | Ground truth homography | Standard evaluation |
| `"cross_image"` | Cross-image validation | Alternative validation |
| `"none"` | No validation | Speed testing only |

### 6. Database Section

**Purpose**: Configure persistent experiment tracking and storage optimization.

```yaml
database:
  connection: "sqlite:///experiments.db"  # Connection string (default)
  save_descriptors: false            # Store raw descriptors (large, usually disabled)
  save_matches: false                # Store match data (large, usually disabled)
  save_visualizations: true          # Store match visualization images
```

**Storage Options**:

| Option | Size Impact | Use Case | Default |
|--------|-------------|----------|---------|
| `save_descriptors` | ⚠️ Very Large | Descriptor analysis/export | `false` |
| `save_matches` | ⚠️ Very Large | Match debugging | `false` |
| `save_visualizations` | 🟡 Moderate | Visual inspection | `true` |

**Best Practices**:
- **Development**: Set all to `false` for faster iteration
- **Production**: Enable `save_visualizations` only
- **Publication**: Enable all for full reproducibility (requires significant storage)

**Note**: Experiment metadata and results are **always stored** regardless of these flags.

## Evaluation Metrics (Bojanic et al. 2020)

DescriptorWorkbench implements the complete evaluation pipeline from **Bojanic et al. (2020): "On the Comparison of Classic and Deep Keypoint Detector and Descriptor Methods"** with three complementary evaluation tasks.

### Image Matching (Default Task)

**Always Enabled**: Standard keypoint matching within image sequences.

**Automatic Metrics**:
- `true_map_micro` / `true_map_macro` - Overall mAP
- `viewpoint_map` / `illumination_map` - HP-V vs HP-I breakdown
- `precision_at_k` / `recall_at_k` - P@K and R@K metrics

**No Configuration Required**: This task runs automatically for all experiments.

### Image Retrieval (Optional)

**Purpose**: Scene-level image retrieval evaluation.

```yaml
evaluation:
  image_retrieval:
    enabled: true                    # Enable image retrieval task
    scorer: "total_matches"          # Scoring method: "total_matches", "map", "precision"
```

**Use Case**: Evaluate descriptor performance for image retrieval applications.

**Metrics Stored**: `image_retrieval_map`, `image_retrieval_queries`

### Keypoint Verification (Bojanic Task 1)

**Purpose**: Test descriptor discriminative power using out-of-sequence distractors.

**Reference**: Bojanic et al. (2020), Section III-B, Equations 1-2
**Baseline**: SIFT+SIFT = 25% AP

```yaml
evaluation:
  keypoint_verification:
    enabled: true                    # Enable verification task (expensive!)
    num_distractor_scenes: 10        # Number of random distractor scenes
    num_distractors_per_scene: 20    # Keypoints sampled per distractor
    seed: 42                         # Random seed for reproducibility
```

**How It Works**:
1. For each keypoint in reference image:
   - Match to **same-sequence images** (positive + hard negatives)
   - Match to **random other-sequence images** (guaranteed negatives/distractors)
2. Rank all candidates by descriptor distance
3. Label using homography ground truth: `y = +1` (correct), `y = -1` (incorrect)
4. Compute Average Precision (AP) over all queries

**Metrics Stored**:
- `keypoint_verification_ap` - Overall verification AP
- `verification_viewpoint_ap` - HP-V (viewpoint changes)
- `verification_illumination_ap` - HP-I (illumination changes)

**Performance Impact**:
- **Computational Cost**: ~5-10x slower than matching
- **Memory**: ~2-5 GB for full HPatches (116 scenes)
- **Typical Runtime**: 15-30 minutes with OpenMP on full dataset

**Parameter Tuning**:
```yaml
# Conservative (faster, for testing)
num_distractor_scenes: 1
num_distractors_per_scene: 10

# Standard (matches verification_test configs)
num_distractor_scenes: 10
num_distractors_per_scene: 20

# Comprehensive (more challenging evaluation)
num_distractor_scenes: 20
num_distractors_per_scene: 50
```

### Keypoint Retrieval (Bojanic Task 3)

**Purpose**: Evaluate ranking quality with three-tier labeling system.

**Reference**: Bojanic et al. (2020), Section III-B, Equations 5-6
**Baseline**: SIFT+SIFT = 26% AP

```yaml
evaluation:
  keypoint_retrieval:
    enabled: true                    # Enable retrieval task (expensive!)
    num_distractor_scenes: 1         # Number of random distractor scenes
    num_distractors_per_scene: 10    # Keypoints sampled per distractor
    seed: 42                         # Random seed for reproducibility
```

**How It Works**:
1. For each keypoint in reference image:
   - Build candidate set from same-sequence + out-of-sequence images
2. Assign three-tier labels:
   - `y = +1`: In-sequence AND closest to homography projection (TRUE_POSITIVE)
   - `y = 0`: In-sequence but NOT closest (HARD_NEGATIVE)
   - `y = -1`: Out-of-sequence (DISTRACTOR)
3. Rank candidates by descriptor distance
4. Compute AP treating **only y=+1 as relevant** (standard IR practice)

**Metrics Stored**:
- `keypoint_retrieval_ap` - Overall retrieval AP
- `retrieval_viewpoint_ap` - HP-V (viewpoint changes)
- `retrieval_illumination_ap` - HP-I (illumination changes)
- `retrieval_num_true_positives` - Count of y=+1 labels
- `retrieval_num_hard_negatives` - Count of y=0 labels (typically very large)
- `retrieval_num_distractors` - Count of y=-1 labels

**Performance Impact**:
- **Computational Cost**: Similar to verification (~5-10x slower than matching)
- **Memory**: ~2-5 GB for full HPatches
- **Typical Runtime**: 20-30 minutes with OpenMP on full dataset

**Expected Label Distribution** (116-scene full run):
- True Positives (y=+1): ~200k (0.03% of candidates)
- Hard Negatives (y=0): ~650M (99.85% of candidates - expected!)
- Distractors (y=-1): ~930k (0.12% of candidates)

**Key Insight**: The massive hard negative count is **correct and expected** - every in-sequence incorrect match counts as a hard negative.

### Running Multiple Metrics

```yaml
evaluation:
  # Standard matching (always enabled)
  matching:
    method: "brute_force"
    norm: "l2"
    cross_check: true
    threshold: 0.8

  # Optional: Image retrieval
  image_retrieval:
    enabled: false

  # Bojanic Task 1: Verification
  keypoint_verification:
    enabled: true
    num_distractor_scenes: 10
    num_distractors_per_scene: 20
    seed: 42

  # Bojanic Task 3: Retrieval
  keypoint_retrieval:
    enabled: true
    num_distractor_scenes: 1
    num_distractors_per_scene: 10
    seed: 42
```

**Total Runtime**: Enabling all three Bojanic metrics (matching + verification + retrieval) takes ~30-40 minutes for full 116-scene evaluation with OpenMP.

### Literature Baselines (Bojanic et al. 2020)

**SIFT+SIFT Baselines**:
- Image Matching: ~59% AP
- Keypoint Verification: ~25% AP (**harder**)
- Keypoint Retrieval: ~26% AP (**harder**)

**Expected Performance Ranking**:
```
Image Matching AP > Keypoint Retrieval AP ≈ Keypoint Verification AP
```

**Our Validation Results**:
- Keypoint Verification: 21.62% AP (vs Bojanic 25% - within acceptable range)
- Keypoint Retrieval: 27.40% AP (vs Bojanic 26% - excellent match! ✅)

## Performance Tuning

**Purpose**: Control parallelization and computational performance.

```yaml
performance:
  parallel_scenes: true              # Enable OpenMP scene-level parallelism
  num_threads: 0                     # Thread count (0 = auto-detect)
  parallel_images: false             # Image-level parallelism (usually false)
  batch_size: 512                    # Batch size for descriptor extraction
  enable_profiling: false            # Enable detailed timing profiling
```

### Performance Options

| Option | Description | Recommended Value | Impact |
|--------|-------------|-------------------|--------|
| `parallel_scenes` | Process scenes in parallel | `true` | 10-16x speedup |
| `num_threads` | OpenMP thread count | `0` (auto) | Auto-detect cores |
| `parallel_images` | Image-level parallelism | `false` | Minimal benefit |
| `batch_size` | Descriptor batch size | `512` | Memory vs speed tradeoff |
| `enable_profiling` | Detailed timing | `false` | Development only |

### Scene-Level Parallelism (Recommended)

**Best for**: Full dataset evaluation with verification/retrieval enabled

```yaml
performance:
  parallel_scenes: true              # Excellent speedup for multi-scene runs
  num_threads: 0                     # Auto-detect (uses all cores)
```

**Speedup Examples**:
- 4 cores: ~3.5x faster
- 8 cores: ~6.5x faster
- 16 cores: ~13x faster

**When to Disable**:
- Single scene evaluation
- Debugging race conditions
- Deterministic profiling

### Thread Count Control

```yaml
# Auto-detect (recommended)
num_threads: 0                       # Uses all available cores

# Manual override (for fair comparisons)
num_threads: 4                       # Force 4 threads
num_threads: 1                       # Disable parallelism
```

### Memory Optimization

**For Large Datasets** (memory-constrained systems):
```yaml
performance:
  parallel_scenes: false             # Sequential processing
  batch_size: 256                    # Smaller batches
database:
  save_descriptors: false            # Disable large storage
  save_matches: false
```

**For Speed** (high-memory systems):
```yaml
performance:
  parallel_scenes: true              # Full parallelism
  batch_size: 1024                   # Larger batches
  num_threads: 0                     # All cores
```

### Profiling Mode

**Development/Debugging Only**:
```yaml
performance:
  enable_profiling: true             # Detailed timing output
  parallel_scenes: false             # Deterministic timing
```

**Output**: Per-scene timing breakdown, descriptor extraction time, matching time, etc.

## Advanced Usage Patterns

### Multi-Descriptor Comparison

```yaml
descriptors:
  # Traditional baseline
  - name: "sift_baseline"
    type: "sift"
    pooling: "none"

  # Enhanced with pooling
  - name: "sift_dsp"
    type: "sift"
    pooling: "domain_size_pooling"
    scales: [0.7, 1.0, 1.4]

  # CNN competitor
  - name: "hardnet_gpu"
    type: "libtorch_hardnet"
    device: "cuda"

  # Fair CPU comparison
  - name: "hardnet_cpu"
    type: "libtorch_hardnet"
    device: "cpu"
```

### Development vs Production Configs

**Development** (fast iteration):
```yaml
dataset:
  scenes: ["i_dome"]                 # Single scene
descriptors:
  - name: "test_descriptor"          # Single descriptor
    type: "sift"
database:
  save_visualizations: false        # Skip outputs
```

**Production** (comprehensive):
```yaml
dataset:
  scenes: []                         # All scenes
descriptors:
  # Multiple descriptors for comparison
database:
  save_visualizations: true          # Persist visual assets
```

## Validation and Error Handling

The system provides detailed validation with specific error messages:

### Common Validation Errors

1. **Missing Required Fields**:
   ```
   Error: Missing required field 'dataset.path'
   ```

2. **Invalid Ranges**:
   ```
   Error: 'ratio_threshold' must be in range [0,1], got 1.5
   ```

3. **Type Mismatches**:
   ```
   Error: Invalid descriptor type 'siftx', valid types: [sift, rgbsift, ...]
   ```

4. **Array Length Mismatches**:
   ```
   Error: 'scale_weights' length (2) must match 'scales' length (3)
   ```

### Validation Rules Summary

- `ratio_threshold`, `stacking_weight`: [0,1]
- `max_features`, `num_octaves`: ≥ 0
- `sigma`, `scale_weight_sigma`: > 0
- `scale_weights` length must match `scales` length
- Descriptor names must be unique
- Required fields: `experiment.name`, `dataset.path`, `descriptor.name`, `descriptor.type`

## Database-First Architecture

> **🎯 KEY INSIGHT**: DescriptorWorkbench uses a **database-only** storage model

### Why Database-Only?

The project migrated from file-based outputs to database-only storage for several reasons:

1. **Better Organization**: All experiment data in one queryable database
2. **Performance**: Faster access and aggregation of results
3. **Consistency**: No file system permissions or path issues
4. **Analysis**: Built-in SQL querying for result comparison
5. **Scalability**: Handles large-scale experiments efficiently

### Accessing Results

Instead of reading CSV files, use SQL queries:

```bash
# Connect to database
sqlite3 experiments.db

# View all experiment results
SELECT descriptor_type, mean_average_precision, processing_time_ms
FROM results
ORDER BY mean_average_precision DESC;

# Compare specific descriptors
SELECT e.descriptor_type, r.mean_average_precision, r.processing_time_ms
FROM experiments e JOIN results r ON e.id = r.experiment_id
WHERE e.descriptor_type IN ('sift', 'libtorch_hardnet')
ORDER BY r.mean_average_precision DESC;

# View experiment configurations
SELECT id, descriptor_type, parameters, timestamp
FROM experiments
WHERE id > 80;
```

### Database Schema

The database stores all experiment data:
- **experiments**: Configuration metadata
- **results**: Performance metrics (MAP, precision, timing)
- **locked_keypoints**: Keypoint coordinates (when enabled)

## Best Practices

### Configuration Design

1. **Use Meaningful Names**:
   ```yaml
   # Good
   name: "sift_dsp_gaussian_0.15"

   # Bad
   name: "test1"
   ```

2. **Start Simple, Add Complexity**:
   ```yaml
   # Start with basic config
   descriptors:
     - name: "sift_baseline"
       type: "sift"

   # Then add advanced features
   # - pooling strategies
   # - device options
   # - multiple descriptors
   ```

3. **Use Comments for Complex Configs**:
   ```yaml
   scales: [0.5, 0.7, 1.0, 1.4, 2.0]    # Wide scale range for texture descriptors
   ratio_threshold: 0.75                  # Stricter matching for noisy scenes
   ```

### Performance Optimization

1. **Development Phase**:
   - Use single scene: `scenes: ["i_dome"]`
   - Single descriptor for debugging
   - Disable file outputs except visualizations
   - Use CPU for deterministic results

2. **Evaluation Phase**:
   - Use appropriate scene selection
   - Enable database tracking
   - Use GPU for CNN descriptors
   - Save visualizations for analysis

3. **Production Phase**:
   - Full scene evaluation: `scenes: []`
   - Multiple descriptor comparison
   - Comprehensive database storage
   - Automated result analysis

### Experiment Design

1. **Fair Comparisons**:
   ```yaml
   # Same device for fair speed comparison
   - name: "sift_cpu"
     device: "cpu"
   - name: "hardnet_cpu"
     device: "cpu"
   ```

2. **Ablation Studies**:
   ```yaml
   # Study pooling effect
   - name: "sift_no_pooling"
     pooling: "none"
   - name: "sift_with_dsp"
     pooling: "domain_size_pooling"
   ```

3. **Parameter Sweeps**:
   ```yaml
   # Different ratio thresholds (create separate configs)
   ratio_threshold: 0.6  # strict_matching.yaml
   ratio_threshold: 0.8  # normal_matching.yaml
   ratio_threshold: 0.9  # loose_matching.yaml
   ```

## Troubleshooting

### Common Issues

1. **"No such file or directory" for dataset**:
   - Check `dataset.path` is correct
   - Ensure HPatches dataset is downloaded
   - Use relative paths from build directory

2. **"Invalid descriptor type"**:
   - Check spelling of descriptor type
   - Ensure required models are available for CNN descriptors

3. **CUDA out of memory**:
   - Reduce `max_features`
   - Use `device: "cpu"` for large datasets
   - Process fewer scenes at once

4. **Empty results**:
   - Check keypoint generation succeeded
   - Verify dataset scenes exist
   - Ensure matching threshold is reasonable

### Debug Strategies

1. **Minimal Configuration**:
   ```yaml
   # Reduce complexity to isolate issues
   dataset: { scenes: ["i_dome"] }
   descriptors: [{ name: "test", type: "sift", pooling: "none" }]
   ```

2. **Enable Verbose Output**:
   ```yaml
   database:
     save_visualizations: true        # Visual debugging
     save_matches: true               # Inspect matcher output
   ```

3. **Check Database Results**:
   ```bash
   sqlite3 experiments.db
   SELECT * FROM results ORDER BY id DESC LIMIT 5;
   ```

### Getting Help

1. Check existing experiment configs in `config/experiments/`
2. Use `reference_comprehensive.yaml` as template
3. Verify your YAML syntax with online validators
4. Check the experiment runner output for specific error messages

## Examples Repository

See `config/experiments/` for working examples:

**Basic Examples**:
- `sift_baseline.yaml` - Simple SIFT evaluation
- `cnn_comparison.yaml` - CNN descriptor comparison
- `pooling_comparison.yaml` - Pooling strategy evaluation

**Bojanic Metrics Examples** (NEW):
- `verification_test.yaml` - Quick 4-scene verification test
- `verification_sift_full.yaml` - Full 116-scene SIFT verification baseline
- `retrieval_test.yaml` - Quick 4-scene retrieval test
- `retrieval_sift_full.yaml` - Full 116-scene SIFT retrieval baseline (all metrics enabled)

**Complete Reference**:
- `reference_comprehensive.yaml` - All options demonstrated (if available)

---

## Version History

**v1.1 (October 2025)**:
- ✅ Added `keypoint_verification` evaluation section
- ✅ Added `keypoint_retrieval` evaluation section
- ✅ Added `performance` section for parallelization control
- ✅ Enhanced `database` section with storage optimization flags
- ✅ Automatic HP-V vs HP-I metric split for all evaluation tasks

**v1.0 (Initial Release)**:
- Core experiment configuration
- Dataset and keypoint setup
- Descriptor configuration with pooling
- Basic evaluation and matching

---

This guide covers the complete YAML configuration system. For additional questions, refer to:
- Source code: `src/core/config/YAMLConfigLoader.cpp`
- Validation tests: `tests/unit/config/`
- Implementation docs: `docs/StatusDocs/METRICS_ENHANCEMENT_PLAN.md`
