# Design: Configurable Matching Methods for Patch Benchmark

**Date:** 2025-03-01
**Status:** Approved
**Author:** Claude + Frank

## Problem

The patch benchmark pipeline uses hardcoded L2 nearest neighbor matching in `MatchingTask::computeSceneMAP()`. This doesn't allow comparing how different matching methods affect mAP, particularly for longer concatenated descriptors where Lowe's ratio test may help filter ambiguous matches.

## Solution

Add configurable matching methods to the patch benchmark, supporting:
- **1-NN (Nearest Neighbor)**: Current behavior, finds single closest match
- **Ratio Test**: Lowe's ratio test, rejects matches where d1/d2 >= threshold

Use OpenCV's `cv::BFMatcher` directly rather than rolling custom implementations.

## Design

### Data Structures

**New enum in `BenchmarkTypes.hpp`:**
```cpp
enum class PatchMatchingMethod {
    NEAREST_NEIGHBOR,  // 1-NN: find single closest match
    RATIO_TEST         // Lowe's ratio test: reject if d1/d2 >= threshold
};
```

**New struct in `BenchmarkTypes.hpp`:**
```cpp
struct MatchingConfig {
    PatchMatchingMethod method = PatchMatchingMethod::NEAREST_NEIGHBOR;
    float ratio_threshold = 0.8f;  // Only used for RATIO_TEST
    int norm_type = cv::NORM_L2;   // L2 for float descriptors
};
```

**Add to existing `Config` struct:**
```cpp
struct Config {
    // ... existing fields ...
    MatchingConfig matching;
};
```

### YAML Configuration

```yaml
tasks:
  matching:
    enabled: true
    method: "nearest_neighbor"  # or "ratio_test"
    ratio_threshold: 0.8        # only used when method is ratio_test
    norm: "l2"                  # l2 (default) or l1
```

### MatchingTask Changes

**Modified `computeSceneMAP()` signature:**
```cpp
static float computeSceneMAP(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const MatchingConfig& matching_config,
    float* accuracy_out = nullptr);
```

**Implementation:**
- Use `cv::BFMatcher` with configured norm type
- For 1-NN: `matcher.match()` returns single best match per query
- For Ratio Test: `matcher.knnMatch()` with k=2, filter by d1/d2 < threshold
- Rejected matches in ratio test count as incorrect for mAP calculation

### File Changes

| File | Change |
|------|--------|
| `src/core/benchmark/BenchmarkTypes.hpp` | Add `PatchMatchingMethod` enum and `MatchingConfig` struct |
| `src/core/benchmark/tasks/MatchingTask.hpp` | Update `computeSceneMAP()` signature |
| `src/core/benchmark/tasks/MatchingTask.cpp` | Replace manual loop with `cv::BFMatcher` |
| `src/cli/patch_benchmark.cpp` | Parse matching config from YAML |

### Defaults & Backward Compatibility

- Default method: `NEAREST_NEIGHBOR` (matches current behavior)
- Default ratio_threshold: `0.8`
- Default norm: `cv::NORM_L2`

Existing configs without `tasks.matching.method` continue to work unchanged.

## Future Extensions

Adding more matchers (FLANN, cross-check variants) requires:
1. Add enum value to `PatchMatchingMethod`
2. Add case in `MatchingTask::computeSceneMAP()`
3. Parse new method name in YAML loader

## Testing

1. Run with `method: nearest_neighbor` - verify results match current behavior
2. Run with `method: ratio_test` - verify different mAP values
3. Compare concatenated descriptor performance between methods
