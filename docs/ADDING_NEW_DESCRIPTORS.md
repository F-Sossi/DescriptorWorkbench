# Adding New Descriptors to DescriptorWorkbench

This guide provides a complete step-by-step process for adding new descriptor types to the DescriptorWorkbench project. The integration has been tested and verified with the addition of an ORB descriptor example.

## Overview

The DescriptorWorkbench uses a modular architecture that makes adding new descriptors straightforward. New descriptors integrate through:

- **Interface-based design**: All descriptors implement `IDescriptorExtractor`
- **Factory pattern**: Centralized creation through `DescriptorFactory`
- **YAML configuration**: String-based descriptor type selection
- **Database integration**: Automatic experiment tracking
- **Build system integration**: CMake handles compilation

## Prerequisites

Before adding a new descriptor, ensure you understand:

- C++ inheritance and virtual functions
- OpenCV descriptor APIs
- CMake build system basics
- YAML configuration structure

## Step-by-Step Integration Process

### 1. Add Descriptor Type to Enum

**File**: `include/thesis_project/types.hpp`

Add your new descriptor to the `DescriptorType` enum:

```cpp
enum class DescriptorType {
    SIFT,                  ///< Standard SIFT descriptor
    HoNC,                  ///< Histogram of Normalized Colors
    RGBSIFT,               ///< RGB color SIFT
    vSIFT,                 ///< Vanilla SIFT implementation
    DSPSIFT,               ///< Domain-Size Pooled SIFT
    VGG,                   ///< VGG descriptor from OpenCV xfeatures2d
    DNN_PATCH,             ///< ONNX-backed patch descriptor via cv::dnn
    LIBTORCH_HARDNET,      ///< LibTorch HardNet CNN descriptor
    LIBTORCH_SOSNET,       ///< LibTorch SOSNet CNN descriptor
    LIBTORCH_L2NET,        ///< LibTorch L2-Net CNN descriptor
    ORB,                   ///< OpenCV ORB binary descriptor  // ← ADD HERE
    YOUR_DESCRIPTOR,       ///< Your new descriptor          // ← ADD HERE
    NONE                   ///< No descriptor
};
```

### 2. Add String Conversion Support

**File**: `include/thesis_project/types.hpp`

Add string conversion in the `toString()` function:

```cpp
inline std::string toString(DescriptorType type) {
    switch (type) {
        case DescriptorType::SIFT: return "sift";
        case DescriptorType::HoNC: return "honc";
        case DescriptorType::RGBSIFT: return "rgbsift";
        case DescriptorType::vSIFT: return "vsift";
        case DescriptorType::DSPSIFT: return "dspsift";
        case DescriptorType::VGG: return "vgg";
        case DescriptorType::DNN_PATCH: return "dnn_patch";
        case DescriptorType::LIBTORCH_HARDNET: return "libtorch_hardnet";
        case DescriptorType::LIBTORCH_SOSNET: return "libtorch_sosnet";
        case DescriptorType::LIBTORCH_L2NET: return "libtorch_l2net";
        case DescriptorType::ORB: return "orb";
        case DescriptorType::YOUR_DESCRIPTOR: return "your_descriptor";  // ← ADD HERE
        case DescriptorType::NONE: return "none";
        default: return "unknown";
    }
}
```

### 3. Create Descriptor Wrapper Implementation

**File**: `src/core/descriptor/extractors/wrappers/YourDescriptorWrapper.hpp`

```cpp
#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv4/opencv2/features2d.hpp>
#include "src/core/config/experiment_config.hpp"

namespace thesis_project::wrappers {

class YourDescriptorWrapper final : public IDescriptorExtractor {
private:
    // Your descriptor implementation (OpenCV or custom)
    cv::Ptr<cv::YourDescriptor> descriptor_;
    std::unique_ptr<experiment_config> config_;

public:
    YourDescriptorWrapper();
    explicit YourDescriptorWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override;

    [[nodiscard]] std::string name() const override { return "YourDescriptor"; }
    [[nodiscard]] int descriptorSize() const override { return 128; }  // Adjust size
    [[nodiscard]] int descriptorType() const override { return CV_32F; }  // or CV_8U for binary

    [[nodiscard]] std::string getConfiguration() const;
};

} // namespace thesis_project::wrappers
```

**File**: `src/core/descriptor/extractors/wrappers/YourDescriptorWrapper.cpp`

```cpp
#include "YourDescriptorWrapper.hpp"
#include <sstream>

namespace thesis_project::wrappers {

YourDescriptorWrapper::YourDescriptorWrapper() {
    descriptor_ = cv::YourDescriptor::create();
}

YourDescriptorWrapper::YourDescriptorWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    // Initialize with parameters from config if needed
    descriptor_ = cv::YourDescriptor::create();
}

cv::Mat YourDescriptorWrapper::extract(const cv::Mat& image,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;

    // Validate keypoints if necessary (example for ORB compatibility)
    for (auto& kp : mutable_keypoints) {
        if (kp.size <= 0.0f || std::isnan(kp.size) || std::isinf(kp.size)) {
            kp.size = 31.0f;  // Set appropriate default size
        }
        // Add other validations as needed
    }

    try {
        descriptor_->compute(image, mutable_keypoints, descriptors);
    } catch (const cv::Exception& e) {
        std::cerr << "YourDescriptor computation failed: " << e.what() << std::endl;
        return cv::Mat();  // Return empty on failure
    }

    return descriptors;
}

std::string YourDescriptorWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "YourDescriptor Wrapper Configuration:\\n";
    ss << "  Descriptor size: " << descriptorSize() << "\\n";
    ss << "  Descriptor type: " << (descriptorType() == CV_32F ? "Float" : "Binary") << "\\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\\n";
    }
    return ss.str();
}

} // namespace thesis_project::wrappers
```

### 4. Update Descriptor Factory

**File**: `src/core/descriptor/factories/DescriptorFactory.cpp`

Add the include:
```cpp
#include "../extractors/wrappers/YourDescriptorWrapper.hpp"
```

Add factory case in `create()` method:
```cpp
std::unique_ptr<IDescriptorExtractor> DescriptorFactory::create(thesis_project::DescriptorType type) {
    switch (type) {
        case thesis_project::DescriptorType::SIFT:
            return createSIFT();
        case thesis_project::DescriptorType::RGBSIFT:
            return createRGBSIFT();
        case thesis_project::DescriptorType::HoNC:
            return std::make_unique<wrappers::HoNCWrapper>();
        case thesis_project::DescriptorType::vSIFT:
            return std::make_unique<wrappers::VSIFTWrapper>();
        case thesis_project::DescriptorType::DSPSIFT:
            return std::make_unique<wrappers::DSPSIFTWrapper>();
        case thesis_project::DescriptorType::VGG:
            return std::make_unique<wrappers::VGGWrapper>();
        case thesis_project::DescriptorType::ORB:
            return std::make_unique<wrappers::ORBWrapper>();
        case thesis_project::DescriptorType::YOUR_DESCRIPTOR:  // ← ADD HERE
            return std::make_unique<wrappers::YourDescriptorWrapper>();
        // ... other cases
        default:
            throw std::runtime_error("Unsupported descriptor type in factory (new-config)");
    }
}
```

Add support case in `isSupported()` method:
```cpp
bool DescriptorFactory::isSupported(thesis_project::DescriptorType type) {
    switch (type) {
        case thesis_project::DescriptorType::SIFT:
        case thesis_project::DescriptorType::RGBSIFT:
        case thesis_project::DescriptorType::HoNC:
        case thesis_project::DescriptorType::vSIFT:
        case thesis_project::DescriptorType::DSPSIFT:
        case thesis_project::DescriptorType::VGG:
        case thesis_project::DescriptorType::ORB:
        case thesis_project::DescriptorType::YOUR_DESCRIPTOR:  // ← ADD HERE
            return true;
        default:
            return false;
    }
}
```

### 5. Update YAML Configuration Loader

**File**: `src/core/config/YAMLConfigLoader.cpp`

Add string-to-enum conversion in `stringToDescriptorType()`:

```cpp
DescriptorType YAMLConfigLoader::stringToDescriptorType(const std::string& str) {
    if (str == "sift") return DescriptorType::SIFT;
    if (str == "rgbsift") return DescriptorType::RGBSIFT;
    if (str == "vsift" || str == "vanilla_sift") return DescriptorType::vSIFT;
    if (str == "honc") return DescriptorType::HoNC;
    if (str == "dnn_patch") return DescriptorType::DNN_PATCH;
    if (str == "vgg") return DescriptorType::VGG;
    if (str == "dspsift") return DescriptorType::DSPSIFT;
    if (str == "libtorch_hardnet") return DescriptorType::LIBTORCH_HARDNET;
    if (str == "libtorch_sosnet") return DescriptorType::LIBTORCH_SOSNET;
    if (str == "libtorch_l2net") return DescriptorType::LIBTORCH_L2NET;
    if (str == "orb") return DescriptorType::ORB;
    if (str == "your_descriptor") return DescriptorType::YOUR_DESCRIPTOR;  // ← ADD HERE
    throw std::runtime_error("Unknown descriptor type: " + str);
}
```

### 6. Update Build System

**File**: `CMakeLists.txt`

Add your wrapper source file to all relevant build targets. Search for patterns like `VGGWrapper.cpp` or `ORBWrapper.cpp` and add your file:

```cmake
# In multiple target_sources commands:
src/core/descriptor/extractors/wrappers/VGGWrapper.cpp
src/core/descriptor/extractors/wrappers/ORBWrapper.cpp
src/core/descriptor/extractors/wrappers/YourDescriptorWrapper.cpp  # ← ADD HERE
```

The CMakeLists.txt has multiple target_sources commands for:
- `experiment_runner`
- `keypoint_manager`
- Test targets (e.g., `test_descriptor_factory_gtest`)

Add your wrapper to each one.

### 7. Create YAML Configuration

**File**: `config/experiments/your_descriptor_test.yaml`

```yaml
experiment:
  name: "your_descriptor_test"
  description: "Test your new descriptor integration"
  version: "1.0"
  author: "your_name"

dataset:
  type: "hpatches"
  path: "../data/"
  scenes: []  # Empty = use all scenes

keypoints:
  generator: "sift"
  max_features: 2000
  contrast_threshold: 0.04
  edge_threshold: 10.0
  source: "homography_projection"
  keypoint_set_name: "sift_homography_projection"
  use_locked_keypoints: false

descriptors:
  - name: "your_descriptor"
    type: "your_descriptor"
    pooling: "none"
    normalize_after_pooling: true

evaluation:
  matching:
    method: "brute_force"
    norm: "l2"  # Use "hamming" for binary descriptors
    cross_check: true
    threshold: 0.8

  validation:
    method: "homography"
    threshold: 0.05
    min_matches: 10

database:
  connection: "sqlite:///experiments.db"
  save_keypoints: false
  save_descriptors: false
  save_matches: false
  save_visualizations: true
```

### 8. Build and Test

```bash
cd build
make -j$(nproc)
./experiment_runner ../config/experiments/your_descriptor_test.yaml
```

## Implementation Notes

### Descriptor Type Considerations

**Float Descriptors (CV_32F)**:
- SIFT, VGG, most CNN descriptors
- Use L2 or cosine distance
- Typically 128-512 dimensions

**Binary Descriptors (CV_8U)**:
- ORB, BRIEF, BRISK
- Use Hamming distance
- Typically 32-256 bits (8-64 bytes)

### Keypoint Compatibility

Different descriptors have different keypoint requirements:

- **SIFT**: Requires scale and orientation information
- **ORB**: Requires valid patch size, computes own orientation
- **CNN**: May require specific patch sizes and normalizations

Add validation in your wrapper's `extract()` method to ensure keypoint compatibility.

### Error Handling

Always include error handling in your descriptor wrapper:

```cpp
try {
    descriptor_->compute(image, mutable_keypoints, descriptors);
} catch (const cv::Exception& e) {
    std::cerr << "Descriptor computation failed: " << e.what() << std::endl;
    return cv::Mat();  // Return empty on failure
}
```

This prevents crashes and allows the experiment to continue with other descriptors.

### Distance Metrics

Choose appropriate distance metrics in your YAML configuration:

- **L2 norm**: Float descriptors (SIFT, VGG, CNN)
- **Hamming**: Binary descriptors (ORB, BRIEF)
- **Cosine**: Normalized float descriptors

## Integration Verification

A successful integration should:

1. ✅ Build without errors
2. ✅ Load YAML configuration correctly
3. ✅ Create descriptor instances via factory
4. ✅ Run experiments without crashes
5. ✅ Store results in database
6. ✅ Handle errors gracefully

## Example Integration: ORB Descriptor

The ORB descriptor has been added as a complete example following this process:

- **Enum**: `DescriptorType::ORB`
- **String**: `"orb"`
- **Wrapper**: `ORBWrapper.hpp/.cpp`
- **Factory**: Support added
- **YAML**: `orb_test.yaml` configuration
- **Build**: CMakeLists.txt updated
- **Testing**: Integration verified

View the ORB implementation files for a complete working example.

## Troubleshooting

### Build Errors
- Check all CMakeLists.txt target_sources have your wrapper
- Verify include paths are correct
- Check for missing dependencies

### Runtime Errors
- Validate keypoint properties in your wrapper
- Add try-catch blocks around descriptor computation
- Check descriptor type and distance metric compatibility

### YAML Errors
- Ensure string conversion is added to YAMLConfigLoader
- Verify YAML syntax is correct
- Check descriptor type string matches enum conversion

## Advanced Features

### Custom Parameters

Add custom parameters to `DescriptorParams` in `types.hpp`:

```cpp
struct DescriptorParams {
    // Existing parameters...

    // Your custom parameters
    int your_param1 = 100;
    float your_param2 = 0.5f;
    std::string your_param3 = "default";
};
```

### Pooling Support

Your descriptor can work with existing pooling strategies:
- **None**: Direct descriptor output
- **Domain-Size Pooling**: Multi-scale aggregation
- **Stacking**: Combination with other descriptors

### Database Integration

Results are automatically stored in SQLite database with:
- Experiment configuration
- Performance metrics (MAP, precision, recall)
- Timing information
- Error logs

## Conclusion

This integration process has been tested and verified. Following these steps ensures your new descriptor will work seamlessly with the DescriptorWorkbench evaluation framework, enabling academic research and performance comparisons.

The modular architecture makes integration straightforward while maintaining compatibility with existing features like pooling, matching, and database tracking.
