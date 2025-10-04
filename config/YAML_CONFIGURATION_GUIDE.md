# DescriptorWorkbench YAML Configuration Guide

This guide provides comprehensive documentation for configuring experiments in the DescriptorWorkbench project using YAML files.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Schema Overview](#schema-overview)
3. [Configuration Sections](#configuration-sections)
4. [Advanced Usage Patterns](#advanced-usage-patterns)
5. [Validation and Error Handling](#validation-and-error-handling)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

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
| `evaluation` | Matching and validation | ✅ |
| `output` | File output settings | ❌ |
| `database` | Database storage settings | ❌ |

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

**Purpose**: Configure persistent experiment tracking.

```yaml
database:
  connection: "sqlite:///experiments.db"  # Connection string
  save_keypoints: true               # Store keypoints in DB
  save_descriptors: false            # Store descriptors in DB
  save_matches: false                # Store matches in DB
  save_visualizations: true          # Store visualizations in DB
```

Database storage is always enabled; toggle individual tables with the `save_*` flags above.

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
- `sift_baseline.yaml` - Simple SIFT evaluation
- `cnn_comparison.yaml` - CNN descriptor comparison
- `pooling_comparison.yaml` - Pooling strategy evaluation
- `reference_comprehensive.yaml` - All options demonstrated

---

This guide covers the complete YAML configuration system. For additional questions, refer to the source code in `src/core/config/` or the validation tests in `tests/unit/config/`.
