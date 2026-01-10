# Keypoint Sets Analysis

This document provides comprehensive statistical analysis of all keypoint sets used in experiments.

## 1. Overview

| Category | Count |
|----------|-------|
| Total keypoint sets | 22 |
| Detector types | 3 (SIFT, SURF, KeyNet) |
| Generation methods | 3 (independent, top_n_size, intersection) |

---

## 2. Keypoint Sets by Detector

### SIFT-Detected Keypoint Sets

| Name | Total | Avg/Image | Min/Image | Max/Image | Avg Scale | Std Scale |
|------|-------|-----------|-----------|-----------|-----------|-----------|
| sift_8000 | 2,553,008 | 3,668 | 143 | 8,000 | 4.45px | 6.72 |
| sift_scale_matched_6px | 645,000 | 927 | 18 | 3,335 | 10.04px | 11.65 |
| sift_surf_8k_a | 645,448 | 927 | 20 | 2,848 | 6.10px | 5.34 |
| sift_surf_scale_matched_intersection_a | 172,909 | 248 | 2 | 796 | 13.29px | 9.68 |
| keynet_sift_8k_b | 565,479 | 813 | 38 | 2,200 | 3.51px | 3.60 |
| sift_keynet_scale_matched_intersection_a | 127,982 | 184 | 6 | 749 | 7.89px | 6.48 |
| sift_scale_only_13px | 400,000 | 575 | - | - | 13.18px | - |
| sift_top_scale_13px | 172,909 | 248 | - | - | 20.77px | - |

### SURF-Detected Keypoint Sets

| Name | Total | Avg/Image | Min/Image | Max/Image | Avg Scale | Std Scale |
|------|-------|-----------|-----------|-----------|-----------|-----------|
| surf_8000 | 2,233,139 | 3,209 | 60 | 8,000 | 30.73px | 24.85 |
| surf_scale_matched_6px | 645,000 | 927 | 18 | 2,546 | 57.09px | 33.11 |
| sift_surf_8k_b | 645,448 | 927 | 20 | 2,848 | 25.48px | 18.65 |
| sift_surf_scale_matched_intersection_b | 172,909 | 248 | 2 | 796 | 50.32px | 28.60 |

### KeyNet-Detected Keypoint Sets

| Name | Total | Avg/Image | Min/Image | Max/Image | Avg Scale | Std Scale |
|------|-------|-----------|-----------|-----------|-----------|-----------|
| keynet_8000 | 2,830,790 | 4,067 | 419 | 7,887 | 49.83px | 29.52 |
| keynet_scale_matched_6px | 645,000 | 927 | 127 | 2,571 | 92.41px | 35.09 |
| keynet_sift_8k_a | 565,479 | 813 | 38 | 2,200 | 45.94px | 25.77 |
| sift_keynet_scale_matched_intersection_b | 127,982 | 184 | 6 | 749 | 90.54px | 33.53 |

---

## 3. Scale Distribution Comparison

### By Detector Type

| Detector | Unfiltered Avg | Filtered Avg | Intersection Avg |
|----------|----------------|--------------|------------------|
| **SIFT** | 4.45px | 10.04px | 13.29px |
| **SURF** | 30.73px | 57.09px | 50.32px |
| **KeyNet** | 49.83px | 92.41px | 90.54px |

**Key Observation:** KeyNet detects features at ~10x larger scales than SIFT. This fundamental difference explains why CNN descriptors (trained for KeyNet scales) struggle on SIFT keypoints.

### Scale Filtering Effect

| Keypoint Set | Before Filtering | After Filtering | Scale Increase |
|--------------|------------------|-----------------|----------------|
| SIFT | 4.45px | 10.04px | **+126%** |
| SURF | 30.73px | 57.09px | **+86%** |
| KeyNet | 49.83px | 92.41px | **+85%** |

### Intersection Effect on Scale

| Intersection | Side A (SIFT) | Side B (Other) | Ratio |
|--------------|---------------|----------------|-------|
| SIFT-SURF | 13.29px | 50.32px | 3.8x |
| SIFT-KeyNet (scale-matched) | 7.89px | 90.54px | 11.5x |
| SIFT-KeyNet (unfiltered) | 3.51px | 45.94px | 13.1x |

**Key Observation:** Even at intersection, detectors produce vastly different scale estimates for the same image locations.

---

## 4. Orientation Analysis

### By Detector Type

| Detector | Avg Angle | Std Angle | Notes |
|----------|-----------|-----------|-------|
| **SIFT** | ~181° | ~104° | Full orientation estimation |
| **SURF** | ~183° | ~105° | Full orientation estimation |
| **KeyNet** | 0° | 0° | **No orientation computed** |

**Critical Finding:** KeyNet does not compute keypoint orientation. All KeyNet keypoints have angle=0. This means:
1. CNN descriptors are not rotation-invariant at the keypoint level
2. Rotation invariance must come from the descriptor or data augmentation

### SIFT Orientation by Keypoint Quality

| Keypoint Set | Avg Angle | Std Angle |
|--------------|-----------|-----------|
| sift_8000 (unfiltered) | 181.01° | 104.24° |
| sift_scale_matched_6px | 182.28° | 104.44° |
| sift_surf_intersection | 183.54° | 104.52° |

Orientation distribution remains consistent regardless of filtering - scale filtering does not bias orientation.

---

## 5. Tolerance Study (SIFT-KeyNet Intersection)

| Tolerance | Total Keypoints | Avg/Image | Min/Image | Max/Image |
|-----------|-----------------|-----------|-----------|-----------|
| 1.0px (strict) | 149,864 | 215 | 1 | 674 |
| 2.0px | 407,988 | 586 | 28 | 1,783 |
| 5.0px | 795,449 | 1,143 | 54 | 2,700 |
| 10.0px (relaxed) | 934,231 | 1,342 | 63 | 3,098 |

### Scale at Different Tolerances (SIFT side)

| Tolerance | Avg Scale | Std Scale |
|-----------|-----------|-----------|
| 1.0px | 2.93px | 2.70 |
| 2.0px | 3.25px | 3.10 |
| 5.0px | 3.76px | 4.45 |
| 10.0px | 4.21px | 5.61 |

**Finding:** Stricter tolerance selects smaller-scale keypoints on the SIFT side (2.93px vs 4.21px). This is counterintuitive but explained by the fact that small-scale SIFT keypoints are more precisely localized.

---

## 6. Response (Strength) Analysis

### By Detector Type

| Detector | Avg Response | Units |
|----------|--------------|-------|
| **SIFT** | 0.032 | Contrast ratio |
| **SURF** | 2,684 - 4,647 | Hessian determinant |
| **KeyNet** | 1,802 - 2,353 | Network score |

**Note:** Response values are not comparable across detectors due to different detection algorithms.

### Response vs Scale Filtering

| SIFT Keypoint Set | Avg Response | Interpretation |
|-------------------|--------------|----------------|
| sift_8000 | 0.0322 | Baseline |
| sift_scale_matched_6px | 0.0338 | +5% stronger |
| sift_surf_intersection | 0.0351 | +9% stronger |

**Finding:** Scale filtering and intersection both select stronger (higher contrast) keypoints.

---

## 7. Per-Image Distribution Analysis

### Images with Fewest Keypoints

| Keypoint Set | Min/Image | Example Scene |
|--------------|-----------|---------------|
| sift_keynet_tol1px | 1 | Extreme cases |
| sift_surf_scale_matched_intersection | 2 | i_fog/6.ppm |
| sift_keynet_scale_matched_intersection | 6 | Challenging illumination |
| sift_scale_matched_6px | 18 | Low-texture scenes |

### Images with Most Keypoints

| Keypoint Set | Max/Image | Example Scene |
|--------------|-----------|---------------|
| sift_8000 | 8,000 | Texture-rich (capped) |
| surf_8000 | 8,000 | Texture-rich (capped) |
| keynet_8000 | 7,887 | Near cap |

---

## 8. Keypoint Set Relationships

### Paired Intersection Sets

| Intersection | Side A | Side B | Relationship |
|--------------|--------|--------|--------------|
| sift_surf_8k | sift_surf_8k_a (SIFT) | sift_surf_8k_b (SURF) | 1-to-1 correspondence |
| sift_surf_scale_matched | sift_surf_scale_matched_intersection_a (SIFT) | sift_surf_scale_matched_intersection_b (SURF) | 1-to-1 correspondence |
| keynet_sift_8k | keynet_sift_8k_a (KeyNet) | keynet_sift_8k_b (SIFT) | 1-to-1 correspondence |
| sift_keynet_scale_matched | sift_keynet_scale_matched_intersection_a (SIFT) | sift_keynet_scale_matched_intersection_b (KeyNet) | 1-to-1 correspondence |

### Derivation Hierarchy

```
sift_8000 (2.5M)
├── sift_scale_matched_6px (645K) - top 25% by size
├── sift_surf_8k_a (645K) - intersection with SURF
│   └── sift_surf_scale_matched_intersection_a (173K) - scale-matched intersection
└── keynet_sift_8k_b (565K) - intersection with KeyNet
    └── sift_keynet_scale_matched_intersection_a (128K) - scale-matched intersection

surf_8000 (2.2M)
├── surf_scale_matched_6px (645K) - top 25% by size
├── sift_surf_8k_b (645K) - intersection with SIFT
│   └── sift_surf_scale_matched_intersection_b (173K) - scale-matched intersection

keynet_8000 (2.8M)
├── keynet_scale_matched_6px (645K) - top 25% by size
├── keynet_sift_8k_a (565K) - intersection with SIFT
│   └── sift_keynet_scale_matched_intersection_b (128K) - scale-matched intersection
```

---

## 9. Summary Statistics

### Keypoint Counts

| Category | Count Range | Notes |
|----------|-------------|-------|
| Unfiltered sets | 2.2M - 2.8M | Near 8K cap per image |
| Scale-filtered sets | 645K | Top 25% by size |
| Intersection (unfiltered) | 565K - 645K | ~20-25% of original |
| Intersection (scale-matched) | 128K - 173K | ~5-7% of original |

### Scale Ranges

| Category | Scale Range |
|----------|-------------|
| SIFT unfiltered | 1.8 - 457px (avg 4.45px) |
| SIFT filtered | 4.2 - 457px (avg 10.04px) |
| SIFT intersection | 4.2 - 439px (avg 13.29px) |
| KeyNet all | 31 - 178px (avg 50-92px) |
| SURF all | 9 - 258px (avg 30-57px) |

---

## 10. Recommendations for Descriptor Evaluation

1. **Always compare on same keypoint set** - Scale differences dramatically affect results

2. **Use scale-matched sets for cross-family comparison** - Reduces confounding factors

3. **CNN descriptors require KeyNet keypoints** - Using SIFT keypoints causes ~60% mAP drop

4. **Intersection sets provide quality filtering** - Fewer but better keypoints

5. **Check min keypoints per image** - Sets with <10 min may have unstable results on some images

6. **KeyNet has no orientation** - Rotation invariance comes from descriptor/augmentation only

---

## 11. Control Sets for Scale vs Intersection Study

Two control keypoint sets were created to isolate the intersection benefit from scale selection:

| Set Name | Count | Avg Scale | Purpose |
|----------|-------|-----------|---------|
| sift_scale_only_13px | 400,000 | 13.18px | Match intersection scale (13.29px) with different count |
| sift_top_scale_13px | 172,909 | 20.77px | Match intersection count (173K) with pure scale selection |

### Key Discovery

Taking the top 173K keypoints by size produces **20.77px average** scale, NOT the intersection's 13.29px. This proves:

1. **Intersection ≠ pure scale selection** - Intersection includes medium-scale keypoints where detectors agree
2. **Detector consensus provides independent quality signal** - Not just larger keypoints

### Experimental Results

| Keypoint Set | Count | Avg Scale | mAP (DSPSIFT) | mAP (RGBSIFT) |
|--------------|-------|-----------|---------------|---------------|
| Intersection | 173K | 13.29px | **74.08%** | **75.03%** |
| Pure Scale | 173K | 20.77px | 70.21% | 71.04% |
| Scale Only | 400K | 13.18px | 67.36% | 67.36% |

**Conclusion:** Detector consensus provides **+7% mAP benefit** independent of scale. See `scale_vs_intersection_study.md` for full analysis.

