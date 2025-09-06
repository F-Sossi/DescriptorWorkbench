# ADR 004: Add VGG Descriptor (Non‑Pooled Comparator)

- Status: Accepted
- Date: 2025-09-01

## Context
We want fair comparisons between classical descriptors (SIFT, DSPSIFT, RGBSIFT, vSIFT) and learned descriptors. VGG (OpenCV contrib) is a widely used handcrafted/learned descriptor that can serve as a non‑pooled baseline. Our pipeline already supports non‑pooled extraction, DSP, stacking, and a DNN patch descriptor.

## Decision
- Add `VGG` descriptor via `cv::xfeatures2d::VGG` as a first‑class `DescriptorType::VGG`.
- Expose via YAML Schema v1 using `type: "vgg"`.
- Implement `VGGWrapper` that conforms to `IDescriptorExtractor`.
- Require `pooling: none` for fair comparisons; rely on L2 matching.
- Detect OpenCV contrib `xfeatures2d` at CMake configure time and define `HAVE_OPENCV_XFEATURES2D` to enable the wrapper. If not present, VGG is disabled and the wrapper throws a clear runtime error if invoked.

## Rationale
- Ensures an apples‑to‑apples comparison by avoiding double pooling (unlike DSPSIFT which pools internally). 
- Keeps the evaluation consistent with existing metrics (True mAP, P@K) and L2 distance.
- Keeps the pipeline modular; VGG is just another extractor behind the common interface.

## Consequences
- Builds without contrib will not expose VGG in `getSupportedTypes()` and will error at runtime if a user forces `type: vgg`.
- Documentation updated to include VGG and its constraints; example config added.

## Alternatives Considered
- Skipping VGG and going straight to neural patch descriptors. We still plan DNN and VGG patch comparators; VGG provides a strong, easy baseline today.

## Implementation Notes
- Files:
  - `src/core/descriptor/extractors/wrappers/VGGWrapper.{hpp,cpp}`
  - `include/thesis_project/types.hpp` (added `DescriptorType::VGG`)
  - `src/core/config/YAMLConfigLoader.cpp` (added YAML mapping `vgg`)
  - `src/core/descriptor/factories/DescriptorFactory.cpp` (factory wiring)
  - `CMakeLists.txt` (detect and define `HAVE_OPENCV_XFEATURES2D`)
- Example config: `config/experiments/vgg_baseline.yaml`

## Next Steps
- Add VGG parameter controls (desc size options) if needed for deeper studies.
- Finalize DNN patch descriptor configs and benchmarking matrix.
