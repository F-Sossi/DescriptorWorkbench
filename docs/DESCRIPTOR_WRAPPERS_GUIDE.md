# Descriptor Wrappers Implementation Guide

This document explains how each descriptor wrapper in `src/core/descriptor/extractors/wrappers/` works and how they implement the `IDescriptorExtractor` interface.

---

## Architecture Overview

All wrappers implement the `IDescriptorExtractor` interface with these core methods:
- `extract(image, keypoints, params)` → Returns descriptor matrix
- `name()` → Returns descriptor name string
- `descriptorSize()` → Returns descriptor dimension
- `descriptorType()` → Returns OpenCV type (CV_32F, CV_8U, etc.)

---

## 1. Traditional Descriptors

### **SIFTWrapper** (`SIFTWrapper.hpp/cpp`)
**Type**: Grayscale gradient-based descriptor
**Dimension**: 128
**Output Type**: CV_32F

**How it works**:
1. Wraps OpenCV's `cv::SIFT::create()` implementation
2. Simple passthrough to `sift_->compute(image, keypoints, descriptors)`
3. No preprocessing required - works on grayscale or color (OpenCV converts internally)

**Key Implementation**:
```cpp
cv::Mat extract(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, ...) {
    cv::Mat descriptors;
    sift_->compute(image, keypoints, descriptors);  // Direct OpenCV call
    return descriptors;
}
```

**Usage**: Standard baseline for grayscale feature matching.

---

### **RGBSIFTWrapper** (`RGBSIFTWrapper.hpp/cpp`)
**Type**: Color-aware SIFT variant
**Dimension**: 384 (3 × 128)
**Output Type**: CV_32F

**How it works**:
1. Wraps custom `RGBSIFT` class from `keypoints/RGBSIFT.h`
2. Computes SIFT descriptors independently on R, G, B channels
3. Concatenates all three 128-dim descriptors into 384-dim vector
4. Requires color image input (BGR format)

**Key Features**:
- Preserves color information lost in grayscale SIFT
- Better for scenes with strong color cues
- 3x memory overhead vs. standard SIFT

**Usage**: Scenes with distinctive color patterns (e.g., logos, painted surfaces).

---

### **VSIFTWrapper** (`VSIFTWrapper.hpp/cpp`)
**Type**: Vanilla SIFT implementation
**Dimension**: 128
**Output Type**: CV_32F

**How it works**:
- Custom implementation of SIFT algorithm from `keypoints/VanillaSIFT.h`
- Provides access to internal SIFT pyramid structure
- Base class for DSP-enabled wrappers (inheritance hierarchy)

**Purpose**: Research-friendly SIFT with accessible internals for experimentation.

---

### **HoNCWrapper** (`HoNCWrapper.hpp/cpp`)
**Type**: Histogram of Normalized Colors
**Dimension**: 128
**Output Type**: CV_32F

**How it works**:
1. Wraps `HoNC` descriptor from `keypoints/HoNC.h`
2. Computes color histograms in normalized RGB space
3. Reduces illumination sensitivity through normalization
4. Requires color image input

**Key Algorithm**:
- Normalizes RGB values: `r' = R/(R+G+B)`, `g' = G/(R+G+B)`
- Builds 2D histogram over (r', g') space
- Flattens histogram to 128-dim descriptor

**Usage**: Robust color matching under varying illumination.

---

### **VGGWrapper** (`VGGWrapper.hpp/cpp`)
**Type**: Learned descriptor from VGG network
**Dimension**: 120 (default)
**Output Type**: CV_32F

**How it works**:
1. Requires OpenCV `xfeatures2d` contrib module (compile-time guard)
2. Wraps `cv::xfeatures2d::VGG::create()`
3. Uses pre-trained VGG network weights

**Conditional Compilation**:
```cpp
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif
```

**Usage**: CNN-learned features with traditional keypoint detection.

---

### **ORBWrapper** (`ORBWrapper.hpp/cpp`)
**Type**: Binary descriptor (Oriented FAST and Rotated BRIEF)
**Dimension**: 32 bytes (256 bits)
**Output Type**: CV_8U (binary)

**How it works**:
1. Wraps OpenCV's `cv::ORB::create()`
2. Computes binary descriptors using intensity comparisons
3. Rotation-invariant through oriented BRIEF

**Key Differences**:
- **Binary descriptor** (not float) → Hamming distance matching required
- Extremely fast computation (no gradients)
- Compact representation (32 bytes vs 128 floats for SIFT)

**Usage**: Real-time applications, mobile devices, SLAM systems.

---

### **SURFWrapper** (`SURFWrapper.hpp/cpp`)
**Type**: Speeded-Up Robust Features
**Dimension**: 64 or 128 (configurable)
**Output Type**: CV_32F

**How it works**:
1. Wraps OpenCV SURF (requires `opencv_contrib`)
2. Uses Haar wavelet responses instead of gradients (faster than SIFT)
3. Approximates Gaussian derivatives with box filters

**Usage**: Faster alternative to SIFT with comparable performance.

---

## 2. Domain-Size Pooling (DSP) Wrappers

### **DSPSIFTWrapper** (`DSPSIFTWrapper.hpp/cpp`)
**Type**: Original DSPSIFT implementation
**Dimension**: 128
**Output Type**: CV_32F
**Performance**: 57.25% mAP (HPatches baseline)

**How it works**:
1. Wraps professor's original `DSPSIFT` class from `keypoints/DSPSIFT.h`
2. Computes SIFT descriptors at multiple scales **during** pyramid construction
3. Samples descriptors from exact Gaussian pyramid built for SIFT detection
4. Aggregates multi-scale descriptors (originally using max pooling)

**Critical Implementation Detail**:
- Descriptor sampling happens **inside** the SIFT pyramid builder
- This ensures scale-space consistency (same pyramid for detection and description)

**Usage**: Reference implementation for DSP performance validation.

---

### **DSPSIFTWrapperV2** (`DSPSIFTWrapperV2.hpp`)
**Type**: Pyramid-aware DSP with configurable aggregation
**Dimension**: 128 (or concatenated if using CONCATENATE)
**Output Type**: CV_32F
**Performance**: 57.25% mAP (matches DSPSIFT exactly)

**How it works** (Template Inheritance Architecture):
```cpp
template<typename SiftType = VanillaSIFT>
class DSPVanillaSIFTWrapper : public SiftType {
    // Inherits protected methods:
    //   - buildGaussianPyramid()
    //   - calcSIFTDescriptor()
    //   - createInitialImage()
};
```

**Key Innovation**:
1. **Inherits from VanillaSIFT** to access internal pyramid methods
2. Builds pyramid once, samples at multiple scales
3. Supports **5 aggregation methods**:
   - `AVERAGE`: Element-wise mean (default)
   - `MAX`: Element-wise maximum (original DSPSIFT)
   - `MIN`: Element-wise minimum
   - `CONCATENATE`: Stack all scales (increases dimension)
   - `WEIGHTED_AVG`: Gaussian/triangular/uniform weighting

**Algorithm Flow**:
```cpp
1. Create base image: createInitialImage(image)
2. Build Gaussian pyramid: buildGaussianPyramid(base)
3. For each scale in {0.85, 1.0, 1.30}:
     - Find pyramid level closest to target scale
     - Extract descriptors at that level
     - Optional: Apply RootSIFT before pooling
4. Aggregate descriptors using selected method
5. Optional: Apply RootSIFT after pooling
6. Normalize (L1 or L2)
```

**Scale Weighting Examples**:
```yaml
# Gaussian weighting (emphasizes center scale)
scale_weighting: "gaussian"
scale_weight_sigma: 0.15

# Explicit weights
scale_weights: [0.3, 0.5, 0.2]  # Sums to 1.0

# Uniform (all scales equal)
scale_weighting: "uniform"
```

**Advantages over External DSP**:
- ✅ Reuses exact SIFT pyramid (no scale-space inconsistency)
- ✅ Configurable aggregation (research flexibility)
- ✅ RootSIFT integration (before or after pooling)
- ✅ Generic template (works with VanillaSIFT, RGBSIFT, HoNC, HoWH)

**Performance Validation**:
```
DSPSIFT (original):     57.25% mAP
DSPSIFT_V2 (MAX):       57.25% mAP ✅ Perfect match
DSPSIFT_V2 (AVERAGE):   57.25% mAP ✅ Perfect match
External DSP (old):     53.36% mAP ❌ 3.89% gap (pyramid mismatch)
```

---

### **DSPRGBSIFTWrapperV2** (`DSPRGBSIFTWrapperV2.hpp/cpp`)
**Type**: Pyramid-aware DSP for RGBSIFT
**Dimension**: 384 (or more with CONCATENATE)
**Output Type**: CV_32F

**How it works**:
1. Template specialization: `DSPVanillaSIFTWrapper<cv::RGBSIFT>`
2. Uses **color Gaussian pyramid** (not grayscale)
3. Computes RGB descriptors at multiple scales
4. Aggregates 384-dim descriptors per scale

**Key Difference from DSPSIFT_V2**:
- Calls `createInitialColorImage()` instead of `createInitialImage()`
- Maintains 3-channel pyramid throughout processing
- Results in 384-dim (or 1152-dim with 3-scale concatenation)

---

### **DSPHoNCWrapperV2** (`DSPHoNCWrapperV2.hpp/cpp`)
**Type**: Pyramid-aware DSP for HoNC descriptor
**Dimension**: 128
**Output Type**: CV_32F

**How it works**:
- Template specialization: `DSPVanillaSIFTWrapper<HoNC>`
- Applies DSP to color histogram descriptors
- Uses color pyramid and HoNC-specific descriptor computation

---

### **DSPHoWHWrapperV2** (`DSPHoWHWrapperV2.hpp/cpp`)
**Type**: Pyramid-aware DSP for HoWH (Histogram of Weighted Hues)
**Dimension**: Variable
**Output Type**: CV_32F

**How it works**:
- Template specialization: `DSPVanillaSIFTWrapper<HoWH>`
- Computes hue-weighted histograms at multiple scales
- Color-aware pyramid processing

---

## 3. Neural Network Descriptors

### **DNNPatchWrapper** (`DNNPatchWrapper.hpp/cpp`)
**Type**: ONNX-based patch descriptor
**Dimension**: Configurable (default 128)
**Output Type**: CV_32F
**Backend**: OpenCV DNN module

**How it works**:

**1. Patch Extraction** (`makePatch_`):
```cpp
Input: Image + Keypoint (pt, size, angle)
↓
1. Calculate scale: S = support_multiplier * kp.size
2. Build affine transform:
   - Rotate: -kp.angle (make upright)
   - Scale: input_size / S
   - Translate: center at (16, 16) for 32×32 patch
3. Warp patch: cv::warpAffine(image, patch, M, 32×32)
4. Normalize:
   - Convert to float [0, 1]
   - Per-patch z-score: (patch - mean) / std
     OR
   - Global normalization: (patch - global_mean) / global_std
↓
Output: 32×32 float tensor
```

**2. Batch Inference**:
```cpp
For each batch of 512 keypoints:
    1. Extract patches → List[32×32 float]
    2. Stack to blob: cv::dnn::blobFromImages() → [B, 1, 32, 32]
    3. Forward pass: net_.forward() → [B, C, 1, 1] or [B, C, H, W]
    4. Handle output shapes:
       - [B, C, 1, 1] → Reshape to [B, C]
       - [B, C, H, W] → Global average pool to [B, C]
    5. Copy to output descriptor matrix
```

**Configuration Options**:
```yaml
dnn:
  model: "hardnet_liberty.onnx"
  input_size: 32                    # Patch size (32×32)
  support_multiplier: 12.0          # Patch window relative to kp.size
  rotate_to_upright: true           # Undo keypoint angle
  mean: 0.0                         # Normalization mean
  std: 1.0                          # Normalization std
  per_patch_standardize: true       # Per-patch z-score (recommended)
```

**Backend Configuration**:
```cpp
// Default: CPU (safe, portable)
net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

// Optional: CUDA acceleration
setBackendTarget(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);
```

**Error Handling**:
- Missing model file → Throws runtime_error
- Invalid ONNX format → Throws runtime_error
- Fallback to `PseudoDNNWrapper` in `experiment_runner.cpp` if ONNX fails

**Usage**: HardNet, SOSNet, L2-Net, or custom trained patch descriptors.

---

### **LibTorchWrapper** (`LibTorchWrapper.hpp/cpp`)
**Type**: PyTorch TorchScript CNN descriptor
**Dimension**: Configurable (default 128)
**Output Type**: CV_32F
**Backend**: LibTorch C++ API

**How it works**:

**1. Model Loading**:
```cpp
Constructor:
    1. Load TorchScript model: torch::jit::load(model_path)
    2. Auto-detect device:
       - CUDA if available → torch::Device(torch::kCUDA, 0)
       - Otherwise CPU → torch::Device(torch::kCPU)
    3. Move model to device: model_.to(device_)
    4. Set evaluation mode: model_.eval()
```

**2. Patch Extraction with Kornia LAF Protocol**:
```cpp
makePatch(imageGray, kp):
    1. Convert keypoint size to radius:
       scale_radius = kp.size / 2.0  // Diameter → radius

    2. Calculate scale using Kornia standard:
       scale = CNN_INPUT_SIZE / (2.0 * scale_radius)
       // Maps radius to half patch size (32/2 = 16 pixels)

    3. Build affine transform (matches Kornia LAF):
       M = getRotationMatrix2D(0, 0, -kp.angle, scale)
       M[0,2] = patch_center_x - kp.pt.x * scale
       M[1,2] = patch_center_y - kp.pt.y * scale

    4. Warp and normalize:
       warpAffine(image, patch, M, 32×32)
       patch = patch.to(float) / 255.0
       if per_patch_standardize:
           patch = (patch - patch.mean()) / patch.std()

    5. Convert to PyTorch tensor: [1, 32, 32]
```

**3. Batch Inference**:
```cpp
extract(image, keypoints, params):
    1. Extract patches for all keypoints → List[Tensor[1, 32, 32]]
    2. Stack to batch: torch::stack() → [N, 1, 32, 32]
    3. Move to device: batch.to(target_device)
    4. Forward pass (no grad):
       output = model_.forward({batch}).toTensor()  // [N, 128]
    5. Move back to CPU: output.to(torch::kCPU)
    6. Convert to OpenCV Mat: tensorToMat(output)
```

**Device Management**:
```cpp
// YAML config overrides constructor device
if (params.device == "cpu") {
    target_device = torch::Device(torch::kCPU);
} else if (params.device == "cuda" && torch::cuda::is_available()) {
    target_device = torch::Device(torch::kCUDA, 0);
}
// "auto" uses constructor-detected device
```

**Configuration Example**:
```yaml
descriptors:
  - name: "hardnet_cnn"
    type: "libtorch_hardnet"
    device: "cuda"  # Override to force GPU
```

**Supported Models**:
- `LIBTORCH_HARDNET`: HardNet descriptor (128-dim)
- `LIBTORCH_SOSNET`: Second-order Similarity Net (128-dim)
- `LIBTORCH_L2NET`: L2-Net descriptor (128-dim)

**Factory Integration** (`LibTorchFactory.cpp`):
```cpp
createHardNet() → Loads "hardnet_pretrained.pt"
createSOSNet() → Loads "sosnet_pretrained.pt"
createL2Net()  → Loads "l2net_pretrained.pt"
```

**Performance Notes**:
- Current performance ceiling: 8.1% mAP (both HardNet/SOSNet)
- Suspected issue: Keypoint detection method (testing KeyNet detector)
- Batch processing (512 patches) for GPU efficiency

---

### **PseudoDNNWrapper** (`PseudoDNNWrapper.hpp/cpp`)
**Type**: Lightweight CNN simulation using traditional CV
**Dimension**: 128
**Output Type**: CV_32F

**How it works** (Fallback for DNNPatchWrapper failures):

**Simulated CNN Architecture**:
```
1. Multi-scale Gaussian Filtering (simulates conv layers)
   - Apply 3 Gaussian kernels with σ = {1.0, 2.0, 3.0}
   - Mimics learned convolutional filters

2. Local Binary Patterns (simulates learned features)
   - Compute LBP on each filtered scale
   - Captures texture patterns (like CNN feature maps)

3. Spatial Pooling (simulates pooling layers)
   - Divide patch into 4×4 grid
   - Compute histogram per cell
   - Concatenate all histograms

4. PCA Dimensionality Reduction (simulates FC layers)
   - Project high-dim features to 128-dim
   - PCA initialized on first batch of patches
```

**Usage**:
- Automatic fallback when ONNX model loading fails
- Documented comparison baseline for DNN descriptors
- Educational tool for understanding CNN-like feature extraction

---

## 4. Wrapper Selection by Factory

**DescriptorFactory** (`DescriptorFactory.cpp`) maps YAML types to wrappers:

```cpp
DescriptorType::SIFT              → SIFTWrapper
DescriptorType::RGBSIFT           → RGBSIFTWrapper
DescriptorType::vSIFT             → VSIFTWrapper
DescriptorType::HoNC              → HoNCWrapper
DescriptorType::DSPSIFT           → DSPSIFTWrapper (original)
DescriptorType::DSPSIFT_V2        → DSPSIFTWrapperV2 (template-based)
DescriptorType::DSPRGBSIFT_V2     → DSPRGBSIFTWrapperV2
DescriptorType::DSPHONC_V2        → DSPHoNCWrapperV2
DescriptorType::DSPHOWH_V2        → DSPHoWHWrapperV2
DescriptorType::VGG               → VGGWrapper
DescriptorType::DNN_PATCH         → DNNPatchWrapper (direct construction in experiment_runner)
DescriptorType::LIBTORCH_HARDNET  → LibTorchFactory::createHardNet()
DescriptorType::LIBTORCH_SOSNET   → LibTorchFactory::createSOSNet()
DescriptorType::LIBTORCH_L2NET    → LibTorchFactory::createL2Net()
DescriptorType::ORB               → ORBWrapper
DescriptorType::SURF              → SURFWrapper
```

---

## 5. Key Design Patterns

### **Template Inheritance (DSP V2 Architecture)**
```cpp
// Base template
template<typename SiftType = VanillaSIFT>
class DSPVanillaSIFTWrapper : public SiftType {
    // Inherits protected pyramid methods
};

// Specializations
DSPVanillaSIFTWrapper<VanillaSIFT>   → DSPSIFT_V2
DSPVanillaSIFTWrapper<cv::RGBSIFT>   → DSPRGBSIFT_V2
DSPVanillaSIFTWrapper<HoNC>          → DSPHONC_V2
DSPVanillaSIFTWrapper<HoWH>          → DSPHOWH_V2
```

**Benefits**:
- Code reuse (single DSP implementation for all SIFT variants)
- Type safety (compile-time checks)
- Access to protected methods (pyramid builders)

### **Adapter Pattern (All Wrappers)**
```cpp
class SomeWrapper : public IDescriptorExtractor {
private:
    cv::Ptr<ActualDescriptor> impl_;  // Wrapped implementation

public:
    cv::Mat extract(...) override {
        // Adapt IDescriptorExtractor interface to underlying implementation
        return impl_->compute(...);
    }
};
```

### **Factory Pattern (Descriptor Creation)**
```cpp
// Central factory hides construction details
auto extractor = DescriptorFactory::create(DescriptorType::SIFT);
```

---

## 6. Performance Comparison Table

| Wrapper | Dimension | Type | HPatches mAP | Speed | Use Case |
|---------|-----------|------|--------------|-------|----------|
| **SIFT** | 128 | Float | 23.0% | Fast | Grayscale baseline |
| **DSPSIFT** | 128 | Float | 57.25% | Medium | Best traditional |
| **DSPSIFT_V2** | 128 | Float | 57.25% | Medium | Research DSP |
| **RGBSIFT** | 384 | Float | TBD | Slow | Color scenes |
| **HoNC** | 128 | Float | TBD | Medium | Illumination changes |
| **VGG** | 120 | Float | TBD | Medium | Learned features |
| **HardNet (LibTorch)** | 128 | Float | 8.1% | Slow | CNN (ceiling) |
| **SOSNet (LibTorch)** | 128 | Float | 8.1% | Slow | CNN (ceiling) |
| **ORB** | 32 bytes | Binary | TBD | Very Fast | Real-time |
| **SURF** | 64-128 | Float | TBD | Fast | SIFT alternative |

---

## 7. Common Implementation Patterns

### **Patch Extraction**
All patch-based descriptors follow this pattern:
```cpp
1. Calculate support window: S = support_mult * kp.size
2. Calculate scale: scale = patch_size / S
3. Build rotation matrix: M = getRotationMatrix2D(kp.pt, -kp.angle, scale)
4. Add translation: M[0,2] += offset_x, M[1,2] += offset_y
5. Warp patch: warpAffine(image, patch, M, patch_size×patch_size)
6. Normalize: Convert to float, z-score standardization
```

### **Batch Processing**
Both DNN wrappers use batching for efficiency:
```cpp
const int BATCH_SIZE = 512;
for (int start = 0; start < total_kps; start += BATCH_SIZE) {
    // Extract batch of patches
    // Run inference on batch
    // Copy results to output matrix
}
```

### **Error Handling**
```cpp
try {
    // Load model / initialize descriptor
} catch (const std::exception& e) {
    throw std::runtime_error("Wrapper initialization failed: " + e.what());
}
```

---

## 8. Adding a New Wrapper

To add a new descriptor wrapper:

1. **Create header/source files**:
   ```
   src/core/descriptor/extractors/wrappers/YourWrapper.hpp
   src/core/descriptor/extractors/wrappers/YourWrapper.cpp
   ```

2. **Implement IDescriptorExtractor**:
   ```cpp
   class YourWrapper : public IDescriptorExtractor {
   public:
       cv::Mat extract(const cv::Mat& image,
                      const std::vector<cv::KeyPoint>& keypoints,
                      const DescriptorParams& params) override;

       std::string name() const override { return "YourDescriptor"; }
       int descriptorSize() const override { return 128; }
       int descriptorType() const override { return CV_32F; }
   };
   ```

3. **Add to types.hpp**:
   ```cpp
   enum class DescriptorType {
       // ...
       YOUR_DESCRIPTOR,
   };
   ```

4. **Add to YAMLConfigLoader**:
   ```cpp
   if (str == "your_descriptor") return DescriptorType::YOUR_DESCRIPTOR;
   ```

5. **Add to DescriptorFactory**:
   ```cpp
   case DescriptorType::YOUR_DESCRIPTOR:
       return std::make_unique<YourWrapper>();
   ```

6. **Update CMakeLists.txt**:
   ```cmake
   add_library(your_wrapper
       src/core/descriptor/extractors/wrappers/YourWrapper.cpp
   )
   ```

---

## Summary

The wrapper architecture provides:
- ✅ **Unified interface** for all descriptor types
- ✅ **Type safety** through templates and enums
- ✅ **Performance** through batch processing and pyramid reuse
- ✅ **Flexibility** through configurable parameters (YAML)
- ✅ **Extensibility** through clean factory pattern

The DSP V2 wrappers represent the state-of-the-art for pyramid-aware multi-scale pooling with zero performance loss compared to the original DSPSIFT implementation.
