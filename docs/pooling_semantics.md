# Pooling Semantics: DSP-SIFT and Stacking

This document explains how Domain-Size Pooling (DSP-SIFT) and Stacking are implemented in this repository and how they relate to the paper “Domain-Size Pooling in Local Descriptors” (Dong & Soatto, arXiv:1412.8556).

## DSP-SIFT Overview

- Goal: Improve invariance of local descriptors (e.g., SIFT) to changes in domain size (scale / support region) without re-detecting keypoints.
- Idea: For each detected keypoint, pool descriptor information over a set of domain sizes centered around the detected support. In practice, compute the descriptor at multiple keypoint sizes and aggregate (average) the results.
- Typical pipeline:
  1) Detect keypoints once (positions remain fixed).
  2) For each domain-size multiplier α in a set A (e.g., α ∈ {0.5, 1.0, 1.5}):
     - Scale the keypoint size (support) by α (do not move the keypoint center).
     - Compute the descriptor on the original image using the enlarged/reduced support.
  3) Aggregate across α (usually an average; optionally weighted).
  4) Normalize the aggregated descriptor row-wise (L2; or RootSIFT: L1 + sqrt).

This reduces aliasing with respect to domain size and increases robustness to blur/scale changes around the support.

## Our Implementation (DomainSizePooling)

- Key design choices (matches DSP intent):
  - Fixed image, variable support: We keep the image unchanged and scale only the keypoint size by α per scale (no image resampling). This preserves geometry, avoids interpolation artifacts, and prevents many out-of-bounds issues.
  - Averaging (not summing): We average descriptors across α (divide by the number of contributing scales or by Σ weights) and then normalize per row. This follows the pooling spirit in DSP-SIFT.
  - Row-wise normalization: Descriptor normalization is applied per row (per keypoint) rather than over the whole matrix.
  - RootSIFT support: When Rooting is enabled, we apply L1 normalization followed by element-wise sqrt per row (before or after pooling depending on configuration).

- Algorithm steps (code path using cv::Feature2D):
  - For each α in `descriptorOptions.scales` (skipping invalid α ≤ 0):
    1) Copy keypoints and set kp.size := kp.size × α.
    2) Compute descriptors on the original image (grayscale if `D_BW`).
    3) Optionally apply per-row normalization (BEFORE_POOLING) and/or RootSIFT (L1 then sqrt).
    4) Accumulate into a running sum (shape-checked on the first contributing scale).
  - After all α:
    5) Average by dividing the sum by the number of contributing scales.
    6) Optionally apply RootSIFT (AFTER_POOLING) and/or per-row normalization (AFTER_POOLING).

- Modern interface:
  - Pooling logic uses `IDescriptorExtractor::extract(image, kps_scaled)` with the same algorithm.

- Shape safety and validation:
  - We require identical row/col/type across scales. If any scale produces a different shape (e.g., differing number of surviving keypoints), we return an empty matrix to signal failure. This keeps the behavior deterministic and debuggable.

- Normalization and Rooting semantics:
  - BEFORE_POOLING:
    - Normalize each per-α descriptor (row-wise, L1 or L2) before pooling.
    - If Rooting is enabled, apply L1 first then sqrt to each row.
  - AFTER_POOLING:
    - Normalize the averaged descriptor (row-wise) once.
    - If Rooting is enabled, apply L1 first then sqrt to each row of the pooled descriptor (optionally followed by L2 if desired).
  - Recommended defaults: Pool raw (or lightly normalized) descriptors and normalize once AFTER pooling.

- Parameterization
  - `descriptorOptions.scales`: set of α multipliers (positive floats). Uniform or log-spaced samples around 1.0 are typical (e.g., {0.5, 1.0, 1.5}).
  - `descriptorOptions.normType`: per-row normalization norm (L1/L2).
  - `descriptorOptions.normalizationStage`: BEFORE_POOLING or AFTER_POOLING.
  - `descriptorOptions.rootingStage`: R_BEFORE_POOLING or R_AFTER_POOLING (RootSIFT).
  - `descriptorOptions.descriptorColorSpace`: D_BW (grayscale) for SIFT, D_COLOR for color descriptors.

- Notes & limitations
- Weighted pooling: Implemented. Either supply explicit `scale_weights` aligned with `scales`, or use procedural weighting via `scale_weighting: gaussian|triangular|uniform` and `scale_weight_sigma`. Gaussian uses log-space distance to α=1.0; triangular linearly tapers with a radius derived from sigma.
  - Out-of-bounds handling: With keypoint-size scaling on the original image, dropouts are significantly reduced. If any scale still yields an incompatible shape, pooling bails to avoid misaligned sums.

## Stacking Overview

- Goal: Combine complementary descriptor information by concatenating two descriptors computed on the same keypoints.
- Pipeline:
  1) Prepare the image for each descriptor’s color-space needs.
  2) Compute descriptor 1 and descriptor 2 on the same (aligned) keypoints.
  3) Optionally normalize (BEFORE_POOLING) each component row-wise, and/or normalize the concatenated descriptor (AFTER_POOLING).
  4) Concatenate horizontally: final dim = dim1 + dim2.

## Our Implementation (Stacking)

- Key design choices:
  - Alignment enforcement: We ensure keypoint positions match between the two computes (within a tolerance). If not aligned, we bail with a clear error (prevents silent row corruption).
  - Normalization: We support per-component normalization/rooting BEFORE concatenation and also normalization/rooting AFTER concatenation. Recommended: row-wise normalization per component before concatenation; optional final normalization after concatenation.

- Steps (legacy detector interface):
  1) Prepare color spaces for both descriptors (BW ↔ COLOR as needed).
  2) Compute descriptors, allowing detectors to adjust keypoints (we capture adjusted keypoints for both).
  3) Ensure row counts match and keypoint positions align (within ε = 0.5 px).
  4) Apply normalization/rooting per configuration.
  5) Concatenate horizontally.

- Notes & limitations
  - Advanced alignment: If detectors drop different rows, a robust intersection/reordering strategy could be added; we currently fail fast to avoid undefined pairing.

## Testing Guidance

- DSP tests to include:
  - Averaging correctness across two α with known descriptors.
  - RootSIFT semantics: L1 + sqrt row-wise, both BEFORE and AFTER pooling.
  - Shape safety: border keypoints at max α either remain valid or pooling fails clearly.
  - Color-space correctness for BW descriptors.

- Stacking tests to include:
  - Alignment enforcement (pass/fail cases).
  - Dimensionality checks (rows, cols) and type.
  - Normalization behavior BEFORE/AFTER concatenation.
  - Color/BW conversion correctness for descriptor pairs.

## References

- Dong, J., & Soatto, S. “Domain-Size Pooling in Local Descriptors.” arXiv:1412.8556
