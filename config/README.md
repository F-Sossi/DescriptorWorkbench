# DescriptorWorkbench Configuration Guide

This directory contains configuration files for DescriptorWorkbench's two evaluation pipelines:

1. **Full Image Pipeline** (`experiment_runner`) - Evaluates descriptors on full HPatches images with keypoint detection
2. **Patch Pipeline** (`patch_benchmark`) - Evaluates descriptors on pre-extracted 65x65 patches

## Directory Structure

```
config/
├── README.md                      # This file
├── YAML_CONFIGURATION_GUIDE.md    # Full image pipeline reference
├── PATCH_BENCHMARK_GUIDE.md       # Patch pipeline reference
├── defaults/                      # Reusable default configurations
├── experiments/                   # Full image pipeline configs
│   ├── *.yaml                     # Active experiment configs
│   └── _archive/                  # Deprecated configs
└── patch_benchmarks/              # Patch pipeline configs
    ├── baselines/                 # Single descriptor baselines
    └── *.yaml                     # Fusion and comparison configs
```

## Pipeline Comparison

| Feature | Full Image (`experiment_runner`) | Patch (`patch_benchmark`) |
|---------|----------------------------------|---------------------------|
| Input | Full HPatches images | Pre-extracted 65x65 patches |
| Keypoint Source | Detected or locked keypoints | Fixed patch centers |
| Measures | Detection + Description | Description only |
| Use Case | End-to-end evaluation | Isolate descriptor quality |
| Speed | Slower (includes detection) | Faster (patches pre-loaded) |

## Quick Start

### Full Image Pipeline

```bash
cd build
./experiment_runner ../config/experiments/thesis_verification_retrieval.yaml
```

### Patch Pipeline

```bash
cd build
./patch_benchmark ../config/patch_benchmarks/patch_sift_full.yaml
```

## Full Image Pipeline (`experiments/`)

Evaluates descriptors on full HPatches images with keypoint detection, matching, and optional verification/retrieval tasks.

### Folder Structure

```
experiments/
├── thesis_*.yaml              # Thesis-ready full evaluations
├── fusion_*.yaml              # Descriptor fusion experiments
├── *_evaluation.yaml          # Single descriptor evaluations
└── _archive/                  # Deprecated/old configs
```

### Configuration Schema

See [YAML_CONFIGURATION_GUIDE.md](YAML_CONFIGURATION_GUIDE.md) for complete documentation.

```yaml
experiment:
  name: "experiment_name"
  description: "Human-readable description"

dataset:
  type: "hpatches"
  path: "../data/"
  scenes: []  # Empty = all scenes

keypoints:
  generator: "sift"
  keypoint_set_name: "sift_keypoints"
  use_locked_keypoints: true

descriptors:
  - name: "sift_baseline"
    type: "sift"
    pooling: "none"
    normalize_after_pooling: true

evaluation:
  matching:
    method: "ratio_test"
    ratio_threshold: 0.8
  keypoint_verification:
    enabled: true
  keypoint_retrieval:
    enabled: true
```

### Keypoint Management

Generate and manage keypoint sets with `keypoint_manager`:

```bash
# Generate SIFT keypoints
./keypoint_manager generate-detector ../data sift sift_keypoints

# Create intersection sets
./keypoint_manager build-intersection \
    --source-a sift_keypoints \
    --source-b keynet_keypoints \
    --out-a sift_keynet_pairs \
    --out-b keynet_sift_pairs \
    --tolerance 5.0

# List available sets
./keypoint_manager list-sets
```

## Patch Pipeline (`patch_benchmarks/`)

Evaluates descriptors on pre-extracted 65x65 HPatches patches, isolating descriptor quality from keypoint detection.

### Folder Structure

```
patch_benchmarks/
├── baselines/                 # Single descriptor baselines
│   ├── sift.yaml
│   ├── hardnet.yaml
│   └── ...
├── patch_sift_full.yaml       # SIFT with all tasks
├── patch_fusion_benchmark.yaml # Descriptor fusion comparison
└── patch_baselines_all.yaml   # All baselines in one run
```

### Configuration Schema

```yaml
experiment:
  name: "patch_benchmark_name"
  description: "Human-readable description"

patches:
  path: "../hpatches-release-rebuilt-color/"
  scenes: []          # Empty = all 116 scenes
  color: true         # Load color patches (required for RGBSIFT, HoNC)
  difficulty:
    easy: true        # ~0.85 overlap
    hard: true        # ~0.72 overlap
    tough: true       # Additional difficulty

tasks:
  mode: "paper"       # Use paper-standard evaluation
  matching:
    enabled: true
  verification:
    enabled: true
    num_positives: 200000
    num_negatives: 1000000
    negative_source: "both"
  retrieval:
    enabled: true
    num_queries: 10000
    num_distractors: 20000
  preload_descriptors: true
  random_seed: 1337

descriptors:
  # Single descriptor
  - name: "sift_baseline"
    type: "sift"
    use_color: false

  # Fusion descriptor
  - name: "sift_hardnet_avg"
    components: ["sift", "libtorch_hardnet"]
    aggregation: "average"
    use_color: false

performance:
  num_threads: 12
  verbose: true

output:
  save_to_database: true
  print_results: true
```

### Descriptor Options

| Type | Dimension | Color Required | Notes |
|------|-----------|----------------|-------|
| `sift` | 128 | No | OpenCV SIFT |
| `rgbsift` | 384 | Yes | 3x128 (R,G,B channels) |
| `rgbsift_channel_avg` | 128 | Yes | Channel-averaged RGBSIFT |
| `honc` | Variable | Yes | Histogram of Normalized Colors |
| `dspsift_v2` | 128 | No | Domain-Size Pooled SIFT |
| `surf` | 64/128 | No | SURF (extended=128) |
| `libtorch_hardnet` | 128 | No | HardNet CNN |
| `libtorch_sosnet` | 128 | No | SOSNet CNN |

### Fusion Aggregation Methods

| Method | Dimension | Requirement |
|--------|-----------|-------------|
| `average` | Same as components | Components must have same dimension |
| `weighted_avg` | Same as components | Components must have same dimension |
| `concatenate` | Sum of components | No dimension requirement |
| `max` | Same as components | Components must have same dimension |
| `min` | Same as components | Components must have same dimension |

**L2 Normalization**: Both pipelines apply L2 normalization before and after fusion to ensure equal contribution from descriptors with different magnitude ranges (e.g., SIFT ~0-512 vs HardNet ~0-1).

## Database Results

Both pipelines store results in `build/experiments.db`:

```bash
# View full image results
sqlite3 experiments.db "SELECT descriptor_type, mean_average_precision FROM results ORDER BY mean_average_precision DESC;"

# View patch benchmark results
sqlite3 experiments.db "SELECT descriptor_name, map_overall, verification_same_overall, retrieval_overall FROM patch_benchmark_results ORDER BY map_overall DESC;"
```

## Evaluation Metrics

Both pipelines implement Bojanic et al. (2020) evaluation:

| Metric | Description | SIFT Baseline |
|--------|-------------|---------------|
| Matching mAP | Image matching precision | ~23% (full) / ~42% (patch) |
| Verification AP | Distractor-based discrimination | ~22% |
| Retrieval AP | Three-tier ranking quality | ~27% |

### HP-V vs HP-I Breakdown

Results automatically split by scene type:
- **HP-V (Viewpoint)**: `v_*` scenes with geometric transformations
- **HP-I (Illumination)**: `i_*` scenes with lighting changes

## Archive

The `experiments/_archive/` folder contains deprecated configuration files from previous organizational schemes. These are kept for reference but should not be used for new experiments.

## Additional Resources

- [YAML_CONFIGURATION_GUIDE.md](YAML_CONFIGURATION_GUIDE.md) - Complete full image pipeline reference
- [skills/run-experiment/](../skills/run-experiment/) - Experiment running guide
- [skills/patch-benchmark/](../skills/patch-benchmark/) - Patch benchmark guide
- [skills/yaml-configuration/](../skills/yaml-configuration/) - YAML configuration help
