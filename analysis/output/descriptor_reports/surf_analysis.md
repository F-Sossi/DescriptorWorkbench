# SURF Descriptor Analysis

This document summarizes all experiments involving SURF and SURF_EXT descriptors.

## 1. Descriptor Overview

**SURF (64D)**: Speeded-Up Robust Features
- Haar wavelet-based descriptor
- Faster than SIFT with comparable accuracy
- Uses integral images for efficiency

**SURF_EXT (128D)**: Extended SURF
- 128-dimensional variant
- Separates positive/negative wavelet responses
- More discriminative but higher dimensional

---

## 2. Baseline Performance by Keypoint Set

### SURF (64D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| surf_8000 (native) | 47.60% | 49.84% | 48.96% | Baseline on SURF keypoints |
| surf_scale_matched_6px | 65.21% | 66.08% | 62.34% | **+17.61%** |
| sift_surf_8k_b (intersection) | 57.86% | 58.99% | 59.02% | SIFT side |
| sift_surf_scale_matched_intersection_b | **75.08%** | 75.28% | 72.68% | **Best SURF** |

### SURF_EXT (128D)

| Keypoint Set | mAP | HP-V | HP-I | Notes |
|--------------|-----|------|------|-------|
| surf_8000 (native) | 43.12% | 44.86% | 45.33% | -4.48% vs SURF |
| surf_scale_matched_6px | 60.85% | 61.17% | 59.42% | -4.36% vs SURF |
| sift_surf_8k_b (intersection) | 53.44% | 54.07% | 55.83% | -4.42% vs SURF |
| sift_surf_scale_matched_intersection_b | 71.18% | 70.88% | 69.92% | -3.90% vs SURF |

---

## 3. Comparison to SIFT Family

### On Comparable Keypoint Sets

| Descriptor | Native 8000 | Scale-Filtered | Intersection |
|------------|-------------|----------------|--------------|
| SURF | 47.60% | 65.21% | 75.08% |
| SURF_EXT | 43.12% | 60.85% | 71.18% |
| SIFT | 42.64% | 63.86% | N/A |
| DSPSIFT_V2 | 46.57% | 65.31% | 74.93% |
| RGBSIFT | 43.61% | 64.69% | 75.03% |

**Finding:** SURF (64D) **outperforms** SIFT (128D) on native keypoints and matches DSPSIFT_V2 on intersection.

### SURF vs SURF_EXT

| Keypoint Set | SURF (64D) | SURF_EXT (128D) | Delta |
|--------------|------------|-----------------|-------|
| surf_8000 | 47.60% | 43.12% | **-4.48%** |
| surf_scale_matched_6px | 65.21% | 60.85% | **-4.36%** |
| intersection | 75.08% | 71.18% | **-3.90%** |

**Finding:** SURF_EXT consistently **underperforms** standard SURF by ~4%. The extended descriptor does not improve matching despite higher dimensionality.

---

## 4. Effect of Keypoint Quality

### Scale Filtering Impact

| Descriptor | Unfiltered | Scale-Filtered | Improvement |
|------------|------------|----------------|-------------|
| SURF | 47.60% | 65.21% | **+17.61%** |
| SURF_EXT | 43.12% | 60.85% | **+17.73%** |

**Finding:** Scale filtering provides ~18% improvement, similar to SIFT family.

### Intersection Set Impact

| Descriptor | Scale-Filtered | Intersection | Improvement |
|------------|----------------|--------------|-------------|
| SURF | 65.21% | 75.08% | **+9.87%** |
| SURF_EXT | 60.85% | 71.18% | **+10.33%** |

**Finding:** Intersection provides additional ~10% improvement.

---

## 5. Fusion Experiments

### SIFT + SURF Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_surf_8k (unfiltered) | concatenate | 61.06% | -0.90% vs RGBSIFT |
| sift_surf_scale_matched_intersection | concatenate | 74.37% | **-0.71% vs SURF** |

**Finding:** SIFT + SURF fusion **underperforms** SURF alone on intersection.

### DSPSIFT + SURF Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_surf_8k (unfiltered) | concatenate | 33.47% | **FAIL** |
| sift_surf_scale_matched_intersection | concatenate | 47.10% | **-27.98% vs SURF** |

**Finding:** DSPSIFT + SURF fusion **fails catastrophically**.

### SIFT + SURF_EXT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_surf_8k | concatenate | 61.06% | Same as SIFT+SURF |
| sift_surf_8k | weighted_avg | 61.06% | Same |
| sift_surf_scale_matched_intersection | concatenate | 74.37% | -0.71% vs SURF |
| sift_surf_scale_matched_intersection | weighted_avg | 74.36% | -0.72% vs SURF |

**Finding:** Using SURF_EXT instead of SURF provides no fusion benefit.

### DSPSIFT + SURF_EXT Fusion

| Keypoint Set | Method | mAP | vs Best Single |
|--------------|--------|-----|----------------|
| sift_surf_8k | concatenate | 25.90% | **FAIL** |
| sift_surf_8k | weighted_avg | 23.31% | **FAIL** |
| sift_surf_scale_matched_intersection | concatenate | 39.53% | **FAIL** |
| sift_surf_scale_matched_intersection | weighted_avg | 36.61% | **FAIL** |

**Finding:** DSPSIFT + SURF_EXT fusion **fails catastrophically**.

### SURF_EXT + RGBSIFT Fusion

| Keypoint Set | Method | mAP | Notes |
|--------------|--------|-----|-------|
| sift_surf_8k | concatenate 512D | 14.52% | **CATASTROPHIC FAIL** |
| sift_surf_scale_matched_intersection | concatenate 512D | 29.28% | **FAIL** |
| sift_surf_8k | RGBSIFT_CHANNEL_AVG concat | 1.13% | **COMPLETE FAIL** |
| sift_surf_scale_matched_intersection | RGBSIFT_CHANNEL_AVG concat | 3.78% | **COMPLETE FAIL** |

**Finding:** SURF_EXT + RGBSIFT fusion produces **catastrophic failures** (1-29% mAP). This suggests severe distribution incompatibility.

---

## 6. Summary Rankings

### Best SURF Configurations

1. **SURF on sift_surf_scale_matched_intersection_b**: 75.08%
2. **SURF_EXT on sift_surf_scale_matched_intersection_b**: 71.18%
3. **SURF on surf_scale_matched_6px**: 65.21%
4. **SURF_EXT on surf_scale_matched_6px**: 60.85%

### Key Conclusions

1. **SURF (64D) outperforms SURF_EXT (128D)** by ~4% consistently
2. **SURF on intersection achieves 75.08% mAP** - matching SIFT-family best
3. **Scale filtering provides ~18% improvement** (similar to SIFT)
4. **SIFT + SURF fusion provides no benefit** - SURF alone is best
5. **DSPSIFT + SURF fusion fails catastrophically** (-28% vs SURF)
6. **SURF_EXT + RGBSIFT fusion fails completely** (~1-3% mAP)

---

## 7. Missing Experiments / Gaps

| Gap | Description | Priority |
|-----|-------------|----------|
| SURF on SIFT keypoints | SURF descriptor on SIFT-detected keypoints | Medium |
| SURF + HoNC fusion | Not tested | Low |
| SURF + CNN fusion | Cross-family with HardNet/SOSNet | Low |

---

## 8. Anomalies and Failures

### DSPSIFT + SURF Catastrophic Failure

The fusion of DSPSIFT with SURF produces extremely poor results:
- Expected: ~70% (average of components)
- Actual: 33-47% on unfiltered/intersection

**Possible causes:**
1. DSP pooling changes descriptor distribution incompatibly with SURF
2. Scale pyramid artifacts interfere with SURF's wavelet responses
3. Normalization incompatibility between pooled SIFT and SURF

### SURF_EXT + RGBSIFT Complete Failure

The most severe failure in the entire experiment set:
- SURF_EXT + RGBSIFT_CHANNEL_AVG: **1.13% mAP** (essentially random)
- SURF_EXT + RGBSIFT (384D): 14.52% mAP

**Hypothesis:** The positive/negative separation in SURF_EXT creates a value distribution incompatible with RGBSIFT's non-negative histograms.

---

## 9. Recommendations

1. **Use standard SURF (64D)** over SURF_EXT - simpler and more accurate
2. **Use SURF only on intersection sets** for best results (75.08%)
3. **Do not fuse SURF with SIFT variants** - provides no benefit
4. **Never fuse SURF_EXT with RGBSIFT** - catastrophic failure
5. **Avoid DSP pooling when using SURF** - causes severe degradation

