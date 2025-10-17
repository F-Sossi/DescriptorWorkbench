# Experiment Configuration Organization

This directory contains organized experiment configurations for the DescriptorWorkbench framework. Each folder represents a specific keypoint set strategy for fair descriptor comparison.

## Folder Structure

```
experiments/
├── Sift_Only/              # Experiments using pure SIFT keypoints
├── Orb_Only/               # Experiments using pure ORB keypoints
├── Surf_Only/              # Experiments using pure SURF keypoints
├── Keynet_Only/            # Experiments using pure KeyNet keypoints
├── Keynet_Sift_Compare/    # Non-overlapping keypoints for CNN vs traditional comparison
├── Orb_Sift_Compare/       # Spatial intersection (5px) between ORB and SIFT
├── Surf_Sift_Compare/      # Spatial intersection (5px) between SURF and SIFT
└── _archive/               # Archived/deprecated experiment files
```

## Required Keypoint Sets

Before running experiments, generate the required keypoint sets using:

```bash
cd /path/to/DescriptorWorkbench
./scripts/generate_all_keypoint_sets.sh
```

### Keypoint Set Definitions

| Keypoint Set Name | Generator | Purpose | Special Properties |
|-------------------|-----------|---------|-------------------|
| `sift_keypoints` | SIFT | Pure SIFT detection | 8000 features, standard params |
| `orb_keypoints` | ORB | Pure ORB detection | 8000 features, native keypoints |
| `surf_keypoints` | SURF | Pure SURF detection | 8000 features, native keypoints |
| `harris_keypoints` | Harris | Harris corner detection | Optional, for additional comparisons |
| `sift_keynet_pairs` | SIFT | CNN vs traditional comparison | 5px spatial intersection with KeyNet |
| `orb_sift_intersection` | ORB+SIFT | Cross-detector comparison | 5px spatial matching tolerance |
| `surf_sift_intersection` | SURF+SIFT | Cross-detector comparison | 5px spatial matching tolerance |

## Experiment Categories

### 1. Detector-Specific Experiments (`*_Only` folders)

These experiments test various descriptor modifications using keypoints from a **single detector**. This ensures the keypoint detection method doesn't influence descriptor performance comparisons.

**Folders:**
- `Sift_Only/` - All SIFT-family descriptors (SIFT, RGBSIFT, HoNC, VSIFT, DSPSIFT variants)
- `Orb_Only/` - ORB descriptor using ORB keypoints
- `Surf_Only/` - SURF descriptor using SURF keypoints

**Use Case:** Compare descriptor modifications (pooling, normalization, rooting) on the same keypoint set.

### 2. CNN vs Traditional Comparison (`Keynet_Sift_Compare/`)

Uses **non-overlapping keypoints** (32px minimum distance) to ensure CNN descriptors don't have overlapping receptive fields, which improves CNN performance.

**Contents:**
- SIFT, RGBSIFT, HoNC descriptor experiments
- HardNet, SOSNet, L2Net CNN baseline experiments
- All using `sift_keynet_pairs` keypoint set

**Use Case:** Fair comparison between CNN and traditional descriptors on the same keypoint locations.

### 3. Cross-Detector Comparisons (`*_Sift_Compare` folders)

Uses **spatial intersection** (5px tolerance) to find keypoints detected by **both detectors**. This tests descriptor robustness when multiple detectors agree on salient points.

**Folders:**
- `Orb_Sift_Compare/` - Keypoints detected by both ORB and SIFT (within 5px)
- `Surf_Sift_Compare/` - Keypoints detected by both SURF and SIFT (within 5px)

**Descriptor Coverage:** The `_Sift_Compare` YAMLs mirror the descriptor menus from their `_Only` counterparts so you can measure how intersection keypoints influence MAP without changing the descriptor parameters.

**Use Case:** Compare descriptors on mutually-agreed salient points from different detectors.

## Running Experiments

### Run Experiments by Folder

```bash
cd build

# Run all SIFT-family experiments on pure SIFT keypoints
./experiment_runner ../config/experiments/Sift_Only/sift_experiments.yaml
./experiment_runner ../config/experiments/Sift_Only/rgbsift_experiments.yaml
./experiment_runner ../config/experiments/Sift_Only/honc_experiments.yaml

# Run CNN vs traditional comparison
./experiment_runner ../config/experiments/Keynet_Sift_Compare/sift_experiments.yaml
./experiment_runner ../config/experiments/Keynet_Sift_Compare/hardnet_baseline.yaml

# Run cross-detector comparison
./experiment_runner ../config/experiments/Orb_Sift_Compare/sift_experiments.yaml
./experiment_runner ../config/experiments/Orb_Sift_Compare/orb_experiments.yaml
./experiment_runner ../config/experiments/Surf_Sift_Compare/sift_experiments.yaml
./experiment_runner ../config/experiments/Surf_Sift_Compare/surf_experiments.yaml
```

### Batch Execution

Create a bash script to run all experiments in a folder:

```bash
#!/bin/bash
for yaml in ../config/experiments/Sift_Only/*.yaml; do
    echo "Running: $yaml"
    ./experiment_runner "$yaml"
done
```

## Experiment Configuration Format

All YAML files follow this structure:

```yaml
experiment:
  name: experiment_name
  description: Human-readable description
  version: '1.0'

keypoints:
  generator: sift|orb|surf|harris
  keypoint_set_name: sift_keypoints  # Must match generated keypoint set
  use_locked_keypoints: true

descriptors:
  - name: descriptor_variant_name
    type: sift|rgbsift|honc|orb|surf|dspsift_v2
    pooling: none|stacking
    normalize_after_pooling: true
    norm_type: 1|4  # L1 or L2
    rooting_stage: none|before_pooling|after_pooling

evaluation:
  matching:
    method: brute_force
    threshold: 0.8
  image_retrieval:
    enabled: false          # Defaults to false; opt in explicitly
    scorer: total_matches   # Alternatives: correct_matches*, ratio_sum

```

### Image Retrieval MAP (Optional)

Setting `evaluation.image_retrieval.enabled: true` triggers a dataset-wide image-level
Mean Average Precision computation. Scorers:

- `total_matches` (default) – rank candidates by surviving match count.
- `ratio_sum` – distance-weighted match votes (`∑ 1/(1+d)`).
- `correct_matches` – only meaningful when using homography-projected keypoints (the
  independent-detection path will report zero because correctness cannot be assessed).

When enabled the runner caches every query/candidate match list in memory so each query
image (typically `1.ppm` per scene) can be ranked against every other image in the dataset.
Results are written to `results.image_retrieval_map`; when the toggle is off the column is
set to `-1` for downstream filtering. Expect extra runtime/memory due to cross-scene
matching.
```

## Keypoint Set Generation Status

### ✅ Implemented CLI Support
- `sift_keypoints` - `./keypoint_manager generate-detector ../data sift sift_keypoints`
- `orb_keypoints` - `./keypoint_manager generate-detector ../data orb orb_keypoints`
- `surf_keypoints` - `./keypoint_manager generate-detector ../data surf surf_keypoints`
- `harris_keypoints` - `./keypoint_manager generate-detector ../data harris harris_keypoints`
- `keynet_keypoints` - `./keypoint_manager generate-kornia-keynet ../data keynet_keypoints 8000 auto --overwrite`
- `sift_keynet_pairs` - `./keypoint_manager build-intersection --source-a sift_keypoints --source-b keynet_keypoints --out-a sift_keynet_pairs --out-b keynet_sift_pairs --tolerance 5.0`

### ✅ Spatial Intersection Support
Use the CLI `build-intersection` workflow to create 5px-overlap keypoint subsets:
```bash
./keypoint_manager build-intersection \
    --source-a sift_keypoints \
    --source-b orb_keypoints \
    --out-a sift_keypoints_orb5px \
    --out-b orb_sift_intersection \
    --tolerance 5.0

./keypoint_manager build-intersection \
    --source-a sift_keypoints \
    --source-b surf_keypoints \
    --out-a sift_keypoints_surf5px \
    --out-b surf_sift_intersection \
    --tolerance 5.0

./keypoint_manager build-intersection \
    --source-a sift_keypoints \
    --source-b keynet_keypoints \
    --out-a sift_keynet_pairs \
    --out-b keynet_sift_pairs \
    --tolerance 5.0
```
The command populates both projected subsets so each detector keeps its native descriptor attributes while sharing the same spatially agreed locations.

## Verification

Check generated keypoint sets:

```bash
cd build
./keypoint_manager list-scenes
sqlite3 experiments.db "SELECT DISTINCT keypoint_set FROM locked_keypoints;"
```

## Database Integration

All experiments store results in `build/experiments.db` with:
- Experiment configurations
- Performance metrics (MAP, precision, recall)
- Keypoint associations

Query results:
```sql
SELECT experiment_name, descriptor_name, map
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE keypoint_set = 'sift_keypoints'
ORDER BY map DESC;
```

## Archive Folder

The `_archive/` folder contains deprecated experiment files from previous organizational schemes. These files are kept for reference but should not be used for new experiments.
