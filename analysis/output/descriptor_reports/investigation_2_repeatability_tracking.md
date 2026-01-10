# Investigation 2: Repeatability Tracking

## Overview

**Goal**: Determine if intersection keypoints are more repeatable (survive image transformations) than non-intersection keypoints.

**Hypothesis**: Keypoints detected by both SIFT and SURF are more likely to be re-detected under image transformations, explaining their superior matching performance.

---

## 1. Background

**Repeatability** measures how consistently a keypoint detector finds the same image locations under transformations:

```
Repeatability = (# keypoints detected in both images at corresponding locations) / (# keypoints in overlap region)
```

For our analysis, we're interested in:
- Do intersection keypoints have higher repeatability than pure-scale keypoints?
- Does repeatability correlate with matching success?

---

## 2. Key Insight from Existing Data

We already have the data to study this! HPatches provides:
- Reference image (image 1)
- Transformed images (images 2-6) with ground truth homographies

**For each keypoint**, we can:
1. Project its location to the transformed image using the homography
2. Check if a keypoint exists at that location in the other image
3. Compute repeatability rates

---

## 3. Experimental Design

### 3.1 Data Available

**Ground Truth Homographies**: `H_1_2`, `H_1_3`, `H_1_4`, `H_1_5`, `H_1_6`

**Keypoint Sets**:
| Set | Count | Purpose |
|-----|-------|---------|
| sift_surf_scale_matched_intersection_a | 173K | Intersection |
| sift_top_scale_13px | 173K | Pure scale control |
| sift_scale_only_13px | 400K | Scale-matched control |

### 3.2 Repeatability Computation

For each keypoint set and each scene:

```python
def compute_repeatability(keypoints_img1, keypoints_img2, H_1_2, tolerance=3.0):
    """
    Compute repeatability between image 1 and image 2.

    Args:
        keypoints_img1: Keypoints in reference image
        keypoints_img2: Keypoints in transformed image
        H_1_2: Homography from image 1 to image 2
        tolerance: Pixel tolerance for correspondence

    Returns:
        repeatability: Fraction of keypoints that are repeated
    """
    repeated = 0
    valid = 0

    for kp1 in keypoints_img1:
        # Project to image 2
        pt1 = np.array([kp1.x, kp1.y, 1.0])
        pt2_proj = H_1_2 @ pt1
        pt2_proj = pt2_proj[:2] / pt2_proj[2]

        # Check if in bounds
        if not in_image_bounds(pt2_proj, img2_size):
            continue

        valid += 1

        # Find nearest keypoint in image 2
        min_dist = float('inf')
        for kp2 in keypoints_img2:
            dist = np.linalg.norm([kp2.x - pt2_proj[0], kp2.y - pt2_proj[1]])
            min_dist = min(min_dist, dist)

        if min_dist < tolerance:
            repeated += 1

    return repeated / valid if valid > 0 else 0
```

### 3.3 Analysis Dimensions

| Dimension | Description |
|-----------|-------------|
| By keypoint set | Compare intersection vs pure scale vs scale only |
| By transformation severity | Image 2 (mild) vs Image 6 (severe) |
| By scene type | HP-V vs HP-I |
| By scale bucket | Tiny, small, medium, large, very large |

---

## 4. Metrics to Compute

### 4.1 Primary Metrics

| Metric | Definition |
|--------|------------|
| Repeatability Rate | % of keypoints repeated within tolerance |
| Mean Localization Error | Average distance to nearest repeated keypoint |
| Scale Consistency | Do repeated keypoints have similar scale? |

### 4.2 Comparative Metrics

| Comparison | Analysis |
|------------|----------|
| Intersection vs Pure Scale | Paired t-test on per-scene repeatability |
| Repeatability vs mAP | Correlation analysis |
| Repeatability by transformation | Trend as transformation increases |

---

## 5. Expected Outcomes

### 5.1 If Intersection Keypoints are More Repeatable

- Higher repeatability rate for intersection set
- Smaller localization error for repeated keypoints
- Repeatability advantage increases with transformation severity
- Strong correlation between repeatability and per-scene mAP

### 5.2 If Repeatability is NOT the Mechanism

- Similar repeatability rates across sets
- Intersection benefit comes from descriptor quality, not detection stability

---

## 6. Implementation Plan

### 6.1 Phase 1: Extract Homographies

**Data location**: `data/hpatches/[scene]/H_1_[2-6]`

```python
def load_homography(scene_path, pair):
    """Load homography matrix for image pair."""
    h_file = os.path.join(scene_path, f'H_1_{pair}')
    H = np.loadtxt(h_file).reshape(3, 3)
    return H
```

### 6.2 Phase 2: Query Keypoints from Database

```sql
-- Get keypoints for a specific scene and image
SELECT x, y, size, angle, response
FROM locked_keypoints k
JOIN keypoint_sets s ON k.keypoint_set_id = s.id
WHERE s.name = 'sift_surf_scale_matched_intersection_a'
  AND k.scene_name = 'v_colors'
  AND k.image_name = '1.ppm';
```

### 6.3 Phase 3: Compute Repeatability

Create `analysis/scripts/analyze_repeatability.py`:

```python
def analyze_repeatability(db_path, keypoint_set_name, data_path):
    """
    Compute repeatability for all scenes and image pairs.

    Returns DataFrame with columns:
    - scene_name
    - pair (2-6)
    - repeatability
    - mean_localization_error
    - num_valid_keypoints
    """
    results = []

    for scene in get_all_scenes(data_path):
        kp_ref = load_keypoints(db_path, keypoint_set_name, scene, '1.ppm')

        for pair in range(2, 7):
            H = load_homography(scene, pair)
            kp_target = load_keypoints(db_path, keypoint_set_name, scene, f'{pair}.ppm')

            rep, loc_error = compute_repeatability(kp_ref, kp_target, H)

            results.append({
                'scene': scene,
                'pair': pair,
                'repeatability': rep,
                'localization_error': loc_error,
                'num_keypoints': len(kp_ref)
            })

    return pd.DataFrame(results)
```

### 6.4 Phase 4: Comparative Analysis

```python
def compare_repeatability(results_intersection, results_pure_scale):
    """Compare repeatability between keypoint sets."""

    # Merge on scene and pair
    merged = results_intersection.merge(
        results_pure_scale,
        on=['scene', 'pair'],
        suffixes=('_int', '_pure')
    )

    # Compute differences
    merged['rep_diff'] = merged['repeatability_int'] - merged['repeatability_pure']

    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(
        merged['repeatability_int'],
        merged['repeatability_pure']
    )

    return merged, t_stat, p_value
```

---

## 7. Visualization Plan

### 7.1 Repeatability Curves

Plot repeatability vs transformation severity (pair 2-6):

```
     Repeatability
100% |----*-----*-----*
     |     \     \     \
 80% |      *-----*-----*  ← Intersection
     |       \     \
 60% |        *-----*-----*  ← Pure Scale
     |
     +----+----+----+----+----+
          2    3    4    5    6
              Image Pair
```

### 7.2 Scatter Plot: Repeatability vs mAP

```
mAP |
85% |                    *
    |              *  *
75% |         *  *
    |      *
65% |   *
    +------------------------
       60%  70%  80%  90%
            Repeatability
```

### 7.3 Box Plots by Scene Type

Separate plots for HP-V and HP-I scenes showing repeatability distributions.

---

## 8. Experiment Configuration

No new experiments needed - analysis uses existing keypoint data and HPatches homographies.

```bash
# Run analysis script
cd /home/frank/repos/DescriptorWorkbench
python3 analysis/scripts/analyze_repeatability.py
```

---

## 9. Questions to Answer

1. Do intersection keypoints have higher repeatability than pure-scale keypoints?
2. Does the repeatability advantage increase with transformation severity?
3. Is there a correlation between per-scene repeatability and per-scene mAP?
4. Do specific scale ranges show different repeatability patterns?
5. Is repeatability more predictive of success in HP-V or HP-I scenes?

---

## 10. Dependencies

- [x] Keypoint data in database (already available)
- [x] HPatches homographies (in data folder)
- [x] Analysis script for repeatability computation
- [x] Visualization scripts

---

## 11. Status

| Phase | Status | Notes |
|-------|--------|-------|
| Design | ✅ Complete | This document |
| Implementation | ✅ Complete | Scripts created |
| Analysis | ✅ Complete | Results below |
| Documentation | ✅ Complete | This update |

---

## 12. EXPERIMENTAL RESULTS

### 12.1 Overall Repeatability

| Keypoint Set | Mean Repeatability | Median | Std Dev | Mean Error |
|--------------|-------------------|--------|---------|------------|
| **Intersection** | **28.02%** | 25.67% | 17.50% | 20.90px |
| Pure Scale | 29.36% | 26.44% | 19.07% | 20.44px |
| Scale Only | 34.40% | 32.64% | 18.57% | 14.10px |

### 12.2 Repeatability by Transformation Severity

| Pair | Intersection | Pure Scale | Scale Only |
|------|--------------|------------|------------|
| 2 (mild) | 39.9% | 43.3% | 47.2% |
| 3 | 31.7% | 32.6% | 37.8% |
| 4 | 25.8% | 27.1% | 32.6% |
| 5 | 23.0% | 23.7% | 29.5% |
| 6 (severe) | 19.7% | 20.2% | 25.0% |

### 12.3 Repeatability by Scene Type

| Set | Viewpoint | Illumination |
|-----|-----------|--------------|
| Intersection | 27.43% | 28.63% |
| Pure Scale | 29.66% | 29.05% |
| Scale Only | 34.48% | 34.31% |

### 12.4 Statistical Comparison

**Intersection vs Pure Scale:**
- Mean difference: **-1.34%** (intersection is LOWER)
- Intersection wins: 255/580 pairs (44.0%)
- Paired t-test: t=-4.277, **p=0.000022** (highly significant)

---

## 13. KEY FINDING: COUNTERINTUITIVE RESULT

**Repeatability is NOT the mechanism for intersection's superior performance!**

| Set | Repeatability | mAP |
|-----|---------------|-----|
| Intersection | 28.02% (lowest) | **74.08%** (highest) |
| Pure Scale | 29.36% | 70.21% |
| Scale Only | 34.40% (highest) | 67.36% (lowest) |

**The relationship is INVERTED:**
- Lower repeatability → Higher mAP
- Higher repeatability → Lower mAP

### 13.1 Interpretation

This counterintuitive result suggests:

1. **Repeatability ≠ matching quality** - A keypoint being re-detected doesn't mean it produces good matches

2. **Descriptor distinctiveness matters more** - Intersection keypoints may produce more distinctive descriptors despite being less repeatable

3. **Quality over quantity** - Intersection selects keypoints that are harder to re-detect but produce better descriptors when they are matched

4. **Detection vs Description separation** - The benefit of intersection comes from the description stage, not the detection stage

### 13.2 Why Lower Repeatability Could Be Better

Possible explanations:
- Intersection excludes "easy" keypoints that are highly repeatable but also highly ambiguous (repetitive textures)
- Intersection selects keypoints with more unique local structure that is harder to exactly re-localize but easier to correctly match
- The scale filtering in intersection (keeping medium scales) may trade off repeatability for distinctiveness

---

## 14. Implications for Future Investigation

This result strongly suggests we should pursue **Investigation 1: NN Ratio Analysis** next.

If intersection keypoints produce more distinctive descriptors (larger NN1/NN2 gap), this would explain:
- Why lower repeatability doesn't hurt matching
- Why intersection achieves higher mAP
- The true mechanism of detector consensus benefit

---

## 15. Scripts and Data

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analysis/scripts/analyze_repeatability.py` | Main repeatability computation |
| `analysis/scripts/plot_repeatability.py` | Visualization generation |

### Output Files

| File | Description |
|------|-------------|
| `logs/repeatability_results.csv` | Detailed per-scene, per-pair results |
| `logs/repeatability_summary.txt` | Summary statistics |
| `logs/repeatability_by_severity.png` | Degradation curves plot |
| `logs/repeatability_vs_map.png` | Repeatability vs mAP scatter |

### Running the Analysis

```bash
cd /home/frank/repos/DescriptorWorkbench
python3 analysis/scripts/analyze_repeatability.py
python3 analysis/scripts/plot_repeatability.py
```

---

## 16. SQL Queries Used

### Query Keypoints for Scene/Image

```sql
SELECT k.x, k.y, k.size, k.angle, k.response
FROM locked_keypoints k
JOIN keypoint_sets s ON k.keypoint_set_id = s.id
WHERE s.name = ?
  AND k.scene_name = ?
  AND k.image_name = ?;
```

### Count Keypoints per Image

```sql
SELECT
    s.name,
    k.scene_name,
    k.image_name,
    COUNT(*) as kp_count
FROM locked_keypoints k
JOIN keypoint_sets s ON k.keypoint_set_id = s.id
WHERE s.name = 'sift_surf_scale_matched_intersection_a'
  AND k.scene_name = 'v_colors'
GROUP BY k.image_name
ORDER BY k.image_name;
```

---

## 17. Related Documents

- `intersection_mechanism_analysis.md` - Parent investigation
- `keypoint_sets_analysis.md` - Keypoint set statistics
- `investigation_1_nn_ratio_analysis.md` - Next priority investigation
- HPatches dataset documentation

---

*Document created: December 2025*
*Results added: December 2025*
