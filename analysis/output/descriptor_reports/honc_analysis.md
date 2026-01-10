# HoNC (Histogram of Normalized Colors) Descriptor Analysis

This document summarizes all experiments involving HoNC and DSPHONC_V2 descriptors.

## 1. Descriptor Overview

**HoNC (128D)**: Histogram of Normalized Colors
- Normalizes RGB: r' = R/(R+G+B), g' = G/(R+G+B)
- Reduces illumination sensitivity through color normalization
- Uses SIFT keypoints (color descriptor on grayscale detector)

**DSPHONC_V2 (128D)**: Domain-Size Pooling variant
- Multi-scale aggregation of HoNC descriptors
- Pyramid-aware pooling strategy

---

## 2. Baseline Performance by Keypoint Set

### HoNC (128D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (unfiltered) | 38.34% | 43.97% | 35.02% | Baseline |
| sift_scale_matched_6px | 58.90% | 63.70% | 50.60% | **+20.56%** |
| sift_surf_scale_matched_intersection_a | **70.64%** | 74.06% | 63.36% | **Best HoNC** |

### DSPHONC_V2 (128D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| sift_8000 (unfiltered) | 41.23% | 47.25% | 36.47% | +2.89% vs HoNC |
| sift_scale_matched_6px | 59.76% | 64.49% | 50.61% | +0.86% vs HoNC |
| sift_surf_scale_matched_intersection_a | 69.45% | 72.57% | 61.37% | **-1.19% vs HoNC** |

---

## 3. Comparison to SIFT Family

### On Same Keypoint Sets

| Keypoint Set | HoNC | SIFT | DSPSIFT_V2 | RGBSIFT |
|--------------|------|------|------------|---------|
| sift_8000 | 38.34% | 42.64% | 46.57% | 43.61% |
| sift_scale_matched_6px | 58.90% | 63.86% | 65.31% | 64.69% |
| sift_surf_intersection | **70.64%** | N/A | 74.93% | 75.03% |

**Finding:** HoNC underperforms SIFT by ~4-5% on same keypoint sets, but the gap narrows on intersection sets.

### Viewpoint vs Illumination

| Descriptor | Keypoint Set | HP-V | HP-I | V-I Delta |
|------------|--------------|------|------|-----------|
| HoNC | sift_8000 | 43.97% | 35.02% | **+8.95%** |
| HoNC | sift_scale_matched_6px | 63.70% | 50.60% | **+13.10%** |
| HoNC | intersection | 74.06% | 63.36% | **+10.70%** |
| SIFT | sift_scale_matched_6px | 65.66% | 59.79% | +5.87% |

**Finding:** HoNC shows **stronger viewpoint invariance** but **weaker illumination invariance** than SIFT. This is unexpected for a color descriptor designed for illumination robustness.

---

## 4. Effect of Keypoint Quality

### Scale Filtering Impact

| Descriptor | Unfiltered | Scale-Filtered | Improvement |
|------------|------------|----------------|-------------|
| HoNC | 38.34% | 58.90% | **+20.56%** |
| DSPHONC_V2 | 41.23% | 59.76% | **+18.53%** |

**Finding:** Scale filtering provides ~20% improvement, similar to SIFT family.

### Intersection Set Impact

| Descriptor | Scale-Filtered | Intersection | Improvement |
|------------|----------------|--------------|-------------|
| HoNC | 58.90% | 70.64% | **+11.74%** |
| DSPHONC_V2 | 59.76% | 69.45% | **+9.69%** |

**Finding:** Intersection provides additional ~10-12% improvement.

### DSP Pooling Effect

| Keypoint Set | HoNC | DSPHONC_V2 | Delta |
|--------------|------|------------|-------|
| sift_8000 | 38.34% | 41.23% | **+2.89%** |
| sift_scale_matched_6px | 58.90% | 59.76% | +0.86% |
| sift_surf_intersection | **70.64%** | 69.45% | **-1.19%** |

**Finding:** DSP helps on unfiltered keypoints but **hurts on intersection sets**. This is the opposite of the SIFT pattern.

---

## 5. Fusion Experiments

### HoNC + SIFT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_8000 | average | 42.67% | = SIFT (42.64%) |
| sift_8000 | concatenate | 42.67% | = SIFT |
| sift_8000 | max | 42.64% | = SIFT |
| sift_scale_matched_6px | average | 63.86% | = SIFT |
| sift_scale_matched_6px | concatenate | 63.86% | = SIFT |
| sift_scale_matched_6px | max | 63.86% | = SIFT |

**Finding:** HoNC + SIFT fusion matches SIFT alone. HoNC provides **no additional value**.

### HoNC + RGBSIFT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_8000 | average | 42.84% | +0.20% vs SIFT |
| sift_8000 | concatenate | 42.84% | +0.20% |
| sift_scale_matched_6px | average | 64.03% | = RGBSIFT_CHANNEL_AVG |
| sift_scale_matched_6px | concatenate | 64.03% | = RGBSIFT_CHANNEL_AVG |

**Finding:** HoNC + RGBSIFT fusion matches RGBSIFT_CHANNEL_AVG. Combining two color descriptors provides **no benefit**.

### HoNC + DSPSIFT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_8000 | average | 45.63% | -0.94% vs DSPSIFT |
| sift_8000 | concatenate | 46.57% | = DSPSIFT |
| sift_8000 | max | 44.79% | -1.78% vs DSPSIFT |
| sift_scale_matched_6px | average | 64.76% | -0.55% vs DSPSIFT |
| sift_scale_matched_6px | concatenate | **65.53%** | +0.22% vs DSPSIFT |
| sift_scale_matched_6px | max | 64.04% | -1.27% vs DSPSIFT |

**Finding:** HoNC + DSPSIFT concatenation provides **marginal improvement** (+0.22%) on scale-filtered sets.

### DSPHONC + SIFT/RGBSIFT Fusion

| Keypoint Set | Fusion Pair | Method | mAP |
|--------------|-------------|--------|-----|
| sift_8000 | DSPHONC + SIFT | average | 42.68% |
| sift_8000 | DSPHONC + RGBSIFT | average | 42.84% |
| sift_scale_matched_6px | DSPHONC + SIFT | average | 63.86% |
| sift_scale_matched_6px | DSPHONC + RGBSIFT | average | 64.03% |

**Finding:** DSPHONC fusion results are nearly identical to HoNC fusion.

### DSPHONC + DSPSIFT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_8000 | average | 46.23% | -1.27% vs DSPRGBSIFT |
| sift_8000 | concatenate | 47.16% | -0.34% vs DSPRGBSIFT |
| sift_scale_matched_6px | average | 64.72% | -1.31% vs DSPRGBSIFT |
| sift_scale_matched_6px | concatenate | **65.41%** | -0.62% vs DSPRGBSIFT |

**Finding:** DSPHONC + DSPSIFT underperforms DSPRGBSIFT_V2 alone (66.03%).

---

## 6. Summary Rankings

### Best HoNC Configurations

1. **HoNC on sift_surf_scale_matched_intersection_a**: 70.64%
2. **DSPHONC_V2 on sift_surf_scale_matched_intersection_a**: 69.45%
3. **DSPHONC_V2 on sift_scale_matched_6px**: 59.76%
4. **HoNC on sift_scale_matched_6px**: 58.90%

### Key Conclusions

1. **HoNC underperforms SIFT** on same keypoint sets by 4-5%
2. **Intersection sets benefit HoNC significantly** (+12% mAP)
3. **DSP pooling hurts HoNC on intersection sets** (opposite of SIFT behavior)
4. **HoNC fusion provides no benefit** - SIFT alone or RGBSIFT alone perform equally or better
5. **HoNC excels on viewpoint changes** but struggles with illumination (unexpected)
6. **Best HoNC result (70.64%)** is still below best SIFT-family result (75.03%)

---

## 7. Missing Experiments / Gaps

| Gap | Description | Priority |
|-----|-------------|----------|
| HoNC on KeyNet intersection | Not tested with CNN detector keypoints | Medium |
| DSPHONC on SIFT-SURF unfiltered intersection | Only tested on scale-matched | Low |
| HoNC + SURF fusion | Not tested | Low |

---

## 8. Recommendations

1. **Use HoNC only on intersection sets** where it approaches SIFT performance
2. **Prefer plain HoNC over DSPHONC_V2** on intersection sets
3. **Do not use HoNC for fusion** - provides no complementary information
4. **For illumination robustness, use RGBSIFT instead** of HoNC

