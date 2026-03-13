#include "PatchTraditionalExtractor.hpp"
#include "core/descriptor/factories/DescriptorFactory.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using thesis_project::factories::DescriptorFactory;

namespace thesis_project {
namespace patches {

PatchTraditionalExtractor::PatchTraditionalExtractor(
    std::unique_ptr<IDescriptorExtractor> base_extractor,
    float keypoint_size,
    bool force_color,
    DescriptorType descriptor_type)
    : base_extractor_(std::move(base_extractor)),
      keypoint_size_(keypoint_size),
      force_color_(force_color),
      descriptor_type_(descriptor_type) {

    if (!base_extractor_) {
        throw std::invalid_argument("PatchTraditionalExtractor: base_extractor cannot be null");
    }
}

cv::Mat PatchTraditionalExtractor::extractFromPatches(
    const std::vector<cv::Mat>& patches,
    const DescriptorParams& params) {

    if (patches.empty()) {
        return cv::Mat();
    }

    cv::Mat all_descriptors;

    // Process each patch individually
    for (const auto& patch : patches) {
        if (patch.empty()) {
            // Skip empty patches, add zero descriptor
            cv::Mat zero_desc = cv::Mat::zeros(1, descriptorSize(), descriptorType());
            all_descriptors.push_back(zero_desc);
            continue;
        }

        // Create synthetic keypoint at patch center
        // For 65x65 patch: center is at (32.5, 32.5)
        const float center_x = static_cast<float>(patch.cols) / 2.0f;
        const float center_y = static_cast<float>(patch.rows) / 2.0f;

        const float kp_size = (params.patch_keypoint_size > 0.0f)
            ? params.patch_keypoint_size
            : keypoint_size_;
        cv::KeyPoint kp(center_x, center_y, kp_size, 0.0f);  // angle = 0 (upright)
        std::vector<cv::KeyPoint> keypoints = {kp};

        cv::Mat input_patch = patch;
        if (force_color_ && patch.channels() == 1) {
            cv::cvtColor(patch, input_patch, cv::COLOR_GRAY2BGR);
        } else if (!force_color_ && patch.channels() == 3) {
            cv::cvtColor(patch, input_patch, cv::COLOR_BGR2GRAY);
        }

        // The base extractor expects an image - treat the patch as a small image
        cv::Mat desc = base_extractor_->extract(input_patch, keypoints, params);

        if (desc.empty() || desc.rows == 0) {
            // Extraction failed, add zero descriptor
            cv::Mat zero_desc = cv::Mat::zeros(1, descriptorSize(), descriptorType());
            all_descriptors.push_back(zero_desc);
        } else {
            all_descriptors.push_back(desc.row(0).clone());
        }
    }

    return all_descriptors;
}

std::unique_ptr<IPatchDescriptorExtractor> PatchTraditionalExtractor::clone() const {
    auto extractor = DescriptorFactory::create(descriptor_type_);
    return std::make_unique<PatchTraditionalExtractor>(
        std::move(extractor),
        keypoint_size_,
        force_color_,
        descriptor_type_);
}

// Factory functions
//
// Default keypoint size = 12.26 for 65x65 patches.
// Derived from the official HPatches benchmark reference implementation:
//   references/hpatches-benchmark/python/extract_opencv_sift.py, line 47:
//     center_kp.size = 2*c/5.303  (where c = patch_size/2 = 32.5)
//     => size = 65 / 5.303 = 12.258 ≈ 12.26
// The constant 5.303 maps OpenCV SIFT's keypoint size to a sampling region that
// fills the 65x65 patch. Larger values (e.g. 41.0) cause SIFT to sample beyond
// the patch boundary, producing degraded descriptors from border padding artifacts.
constexpr float kDefaultPatchKeypointSize = 12.26f;

std::unique_ptr<IPatchDescriptorExtractor> createPatchSIFT() {
    auto sift = DescriptorFactory::create(DescriptorType::SIFT);
    return std::make_unique<PatchTraditionalExtractor>(std::move(sift), kDefaultPatchKeypointSize, false, DescriptorType::SIFT);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFT() {
    auto rgbsift = DescriptorFactory::create(DescriptorType::RGBSIFT);
    return std::make_unique<PatchTraditionalExtractor>(std::move(rgbsift), kDefaultPatchKeypointSize, true, DescriptorType::RGBSIFT);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFTChannelAvg() {
    auto rgbsift_avg = DescriptorFactory::create(DescriptorType::RGBSIFT_CHANNEL_AVG);
    return std::make_unique<PatchTraditionalExtractor>(std::move(rgbsift_avg), kDefaultPatchKeypointSize, true, DescriptorType::RGBSIFT_CHANNEL_AVG);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchHoNC() {
    auto honc = DescriptorFactory::create(DescriptorType::HoNC);
    return std::make_unique<PatchTraditionalExtractor>(std::move(honc), kDefaultPatchKeypointSize, true, DescriptorType::HoNC);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchDSPSIFT() {
    auto dspsift = DescriptorFactory::create(DescriptorType::DSPSIFT_V2);
    // NOTE: DSP (Domain Size Pooling) is designed for scale variation in full images.
    // On pre-extracted 65x65 patches that are already scale-normalized, DSP multi-scale
    // pooling doesn't provide meaningful benefit. For proper DSP evaluation, use the
    // full image experiment_runner pipeline.
    return std::make_unique<PatchTraditionalExtractor>(std::move(dspsift), kDefaultPatchKeypointSize, false, DescriptorType::DSPSIFT_V2);
}

// SURF keypoint size for 65x65 patches.
// Derived from OpenCV SURF source (opencv_contrib/modules/xfeatures2d/src/surf.cpp):
//   s = size * 1.2 / 9.0
//   win_size = (PATCH_SZ + 1) * s = 21 * size * 1.2 / 9.0 = 2.8 * size
// The window extends win_size/2 from center, must fit within 32.5px (half of 65):
//   size <= 65 / 2.8 = 23.21
constexpr float kSurfPatchKeypointSize = 23.21f;

std::unique_ptr<IPatchDescriptorExtractor> createPatchSURF() {
    auto surf = DescriptorFactory::create(DescriptorType::SURF);
    return std::make_unique<PatchTraditionalExtractor>(std::move(surf), kSurfPatchKeypointSize, false, DescriptorType::SURF);
}

} // namespace patches
} // namespace thesis_project
