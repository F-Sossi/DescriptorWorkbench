# Investigation 4: False Positive Forensics

## Overview

**Goal**: Examine actual wrong matches to understand what types of errors occur and whether intersection keypoints produce different error patterns.

**Hypothesis**: Intersection keypoints may produce fewer "confusing" descriptors, reducing specific types of false matches (e.g., repetitive texture errors, scale confusion).

---

## 1. Background

Matching errors can be categorized:

| Error Type | Description | Example |
|------------|-------------|---------|
| **Repetitive texture** | Match to similar but wrong location | Brick wall, text, patterns |
| **Scale confusion** | Match at wrong scale level | Multi-scale features |
| **Geometric ambiguity** | Similar local appearance | Corners, edges |
| **Illumination sensitivity** | Descriptor changes with lighting | Shadows, highlights |
| **Occlusion** | Keypoint not visible in target | Behind objects |

**Question**: Does intersection filtering reduce specific error types?

---

## 2. Data Requirements

### 2.1 What We Need

| Data | Purpose |
|------|---------|
| Match results | Which keypoints were matched |
| Ground truth | Correct correspondences via homography |
| Descriptor distances | NN1/NN2 for each match |
| Keypoint properties | Location, scale, response |
| Image patches | Visual inspection of errors |

### 2.2 Database Support

Current schema has `matches` table:

```sql
PRAGMA table_info(matches);
```

Need to verify what's stored and potentially add:
- Match distance (L2)
- Is correct match (boolean)
- Projected ground truth location

---

## 3. Error Taxonomy

### 3.1 Categories for Analysis

| Category | Detection Method |
|----------|------------------|
| **Near-miss** | Matched within 10px of correct, but outside 3px tolerance |
| **Same-object** | Matched to different part of same object |
| **Repetitive** | Matched to similar texture elsewhere |
| **Scale-related** | Matched keypoint has very different scale |
| **Random** | No obvious pattern, large distance |

### 3.2 Quantification

For each error category, compute:
- Frequency (% of total errors)
- Average descriptor distance
- Average spatial error
- Prevalence by scene type

---

## 4. Experimental Design

### 4.1 Sample Selection

Focus on scenes with large intersection benefit or deficit:

**High intersection benefit:**
- v_colors (+31.6%)
- i_dome (+13.7%)
- i_fog (+12.7%)

**Pure scale wins:**
- i_brooklyn (-11.9%)
- i_londonbridge (-10.6%)

### 4.2 Analysis Pipeline

```python
def analyze_false_positives(matches, ground_truth_H, keypoints_ref, keypoints_target):
    """
    Categorize false positive matches.

    Returns: DataFrame with error analysis
    """
    errors = []

    for match in matches:
        kp_ref = keypoints_ref[match.query_idx]
        kp_match = keypoints_target[match.train_idx]

        # Compute ground truth location
        pt_ref = np.array([kp_ref.x, kp_ref.y, 1.0])
        pt_gt = ground_truth_H @ pt_ref
        pt_gt = pt_gt[:2] / pt_gt[2]

        # Compute error
        error_dist = np.linalg.norm([kp_match.x - pt_gt[0], kp_match.y - pt_gt[1]])

        if error_dist > 3.0:  # False positive threshold
            error_type = categorize_error(
                kp_ref, kp_match, pt_gt, error_dist,
                keypoints_target, image_ref, image_target
            )

            errors.append({
                'ref_x': kp_ref.x, 'ref_y': kp_ref.y,
                'match_x': kp_match.x, 'match_y': kp_match.y,
                'gt_x': pt_gt[0], 'gt_y': pt_gt[1],
                'error_dist': error_dist,
                'error_type': error_type,
                'scale_ref': kp_ref.size,
                'scale_match': kp_match.size,
                'match_distance': match.distance
            })

    return pd.DataFrame(errors)
```

### 4.3 Error Categorization Logic

```python
def categorize_error(kp_ref, kp_match, pt_gt, error_dist,
                      keypoints_target, img_ref, img_target):
    """
    Categorize a false positive match.
    """
    # Near-miss: Close but not quite
    if error_dist < 10:
        return 'near_miss'

    # Scale confusion: Large scale difference
    scale_ratio = kp_match.size / kp_ref.size
    if scale_ratio > 2.0 or scale_ratio < 0.5:
        return 'scale_confusion'

    # Check for repetitive texture
    # (Would require patch comparison)
    if is_repetitive_region(kp_ref, img_ref):
        return 'repetitive_texture'

    # Check if matched to different part of same object
    # (Semantic analysis - simplified)
    if error_dist < 50:
        return 'same_object'

    return 'random'
```

---

## 5. Metrics to Compute

### 5.1 Error Distribution

| Metric | Intersection | Pure Scale | Difference |
|--------|--------------|------------|------------|
| % near-miss | ? | ? | ? |
| % scale confusion | ? | ? | ? |
| % repetitive | ? | ? | ? |
| % random | ? | ? | ? |

### 5.2 Per-Scene Analysis

For high-gain scenes (v_colors, i_dome):
- What error types did intersection eliminate?
- Visual examples of prevented errors

For low-gain scenes (i_brooklyn):
- What error types increased?
- Why did intersection not help?

---

## 6. Implementation Plan

### 6.1 Phase 1: Enable Match Storage

**Verify match storage is enabled:**
```yaml
database:
  save_matches: true
```

**Check matches table structure:**
```sql
PRAGMA table_info(matches);
```

### 6.2 Phase 2: Run Targeted Experiments

Run experiments on selected scenes with match storage:

```yaml
# config/experiments/false_positive_study.yaml
experiment:
  name: "false_positive_analysis"
  description: "Store matches for error analysis"

dataset:
  type: "hpatches"
  path: "../data/"
  scenes:
    - "v_colors"
    - "i_dome"
    - "i_fog"
    - "i_brooklyn"
    - "i_londonbridge"

database:
  save_matches: true
  save_descriptors: false  # Keep storage manageable
```

### 6.3 Phase 3: Analysis Script

Create `analysis/scripts/analyze_false_positives.py`:

```python
def main():
    # 1. Load matches from database
    matches_int = load_matches(db, 'intersection')
    matches_pure = load_matches(db, 'pure_scale')

    # 2. Load ground truth homographies
    homographies = load_homographies(data_path)

    # 3. Categorize errors
    errors_int = analyze_false_positives(matches_int, homographies)
    errors_pure = analyze_false_positives(matches_pure, homographies)

    # 4. Compare distributions
    compare_error_distributions(errors_int, errors_pure)

    # 5. Generate visualizations
    visualize_error_examples(errors_int, errors_pure, images)
```

### 6.4 Phase 4: Visual Inspection

Generate montages of false positive examples:

```python
def create_error_montage(errors, images, output_path):
    """
    Create visual montage showing false positive matches.

    Each row: [ref_patch] [match_patch] [gt_patch] [error_type]
    """
    for i, error in enumerate(errors[:20]):  # Top 20 examples
        ref_patch = extract_patch(images['ref'], error['ref_x'], error['ref_y'])
        match_patch = extract_patch(images['target'], error['match_x'], error['match_y'])
        gt_patch = extract_patch(images['target'], error['gt_x'], error['gt_y'])

        # Combine into row
        row = np.hstack([ref_patch, match_patch, gt_patch])
        # Add to montage
```

---

## 7. Visualization Plan

### 7.1 Error Type Bar Chart

```
                     Intersection    Pure Scale
Near-miss       |████████          |██████████████
Scale confusion |████              |████████████
Repetitive      |██████            |████████████████
Random          |████████████      |████████████████
                +------------------+------------------
                0%       20%      0%       20%
```

### 7.2 Error Examples Montage

| Query Patch | Matched Patch | Correct Patch | Error Type |
|-------------|---------------|---------------|------------|
| [image] | [image] | [image] | repetitive |
| [image] | [image] | [image] | scale_confusion |

### 7.3 Spatial Error Heatmap

Show where errors occur in the image:
- Overlay on reference image
- Color by error type
- Compare intersection vs pure scale

---

## 8. Experiment Configuration

```yaml
# config/experiments/false_positive_study.yaml
experiment:
  name: "false_positive_forensics"
  description: "Store matches for error analysis on selected scenes"

dataset:
  type: "hpatches"
  path: "../data/"
  scenes:
    # High intersection benefit
    - "v_colors"
    - "i_dome"
    - "i_fog"
    - "i_school"
    - "i_santuario"
    # Pure scale wins
    - "i_brooklyn"
    - "i_londonbridge"
    - "i_indiana"
    - "i_crownday"
    - "i_books"

keypoints:
  source: "database"
  use_locked_keypoints: true

descriptors:
  - name: "dspsift_v2__intersection__fp_study"
    type: "dspsift_v2"
    keypoint_set_name: "sift_surf_scale_matched_intersection_a"
    pooling: "none"

  - name: "dspsift_v2__pure_scale__fp_study"
    type: "dspsift_v2"
    keypoint_set_name: "sift_top_scale_13px"
    pooling: "none"

evaluation:
  matching:
    method: "ratio_test"
    ratio_threshold: 0.8

database:
  connection: "sqlite:///experiments.db"
  save_matches: true
  save_descriptors: false
```

---

## 9. Questions to Answer

1. What types of errors are most common for each keypoint set?
2. Does intersection reduce specific error types (repetitive, scale confusion)?
3. Are there error types that increase with intersection?
4. Do high-gain scenes have different error profiles than low-gain scenes?
5. Can we predict intersection benefit from error type distribution?
6. Are errors correlated with keypoint properties (scale, response)?

---

## 10. Dependencies

- [ ] Verify matches table schema
- [ ] Match storage enabled in experiments
- [ ] Ground truth homography loading
- [ ] Error categorization logic
- [ ] Patch extraction utilities
- [ ] Visualization scripts

---

## 11. Status

| Phase | Status | Notes |
|-------|--------|-------|
| Design | ✅ Complete | This document |
| Schema check | ⬜ Not started | Verify matches table |
| Experiments | ⬜ Not started | Run with match storage |
| Analysis | ⬜ Not started | |
| Visualization | ⬜ Not started | |
| Documentation | ⬜ Not started | |

---

## 12. Quick Start

Check matches table structure:

```bash
sqlite3 build/experiments.db "PRAGMA table_info(matches);"
sqlite3 build/experiments.db "SELECT COUNT(*) FROM matches;"
```

---

## 13. Related Documents

- `intersection_mechanism_analysis.md` - Parent investigation
- `investigation_1_nn_ratio_analysis.md` - Descriptor distinctiveness
- `investigation_2_repeatability_tracking.md` - Keypoint stability

---

## 14. Expected Insights

This investigation could reveal:

1. **Repetitive texture immunity**: If intersection reduces repetitive texture errors, it explains why v_colors (+31.6%) benefits so much - the image has strong color patterns.

2. **Scale confusion resistance**: If intersection reduces scale errors, it validates that multi-detector consensus provides scale-stable keypoints.

3. **Scene-specific patterns**: Different scenes may have different dominant error types, explaining variable intersection benefit.

---

*Document created: December 2025*
