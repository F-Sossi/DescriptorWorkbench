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

        cv::KeyPoint kp(center_x, center_y, keypoint_size_, 0.0f);  // angle = 0 (upright)
        std::vector<cv::KeyPoint> keypoints = {kp};

        cv::Mat input_patch = patch;
        if (force_color_ && patch.channels() == 1) {
            cv::cvtColor(patch, input_patch, cv::COLOR_GRAY2BGR);
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
std::unique_ptr<IPatchDescriptorExtractor> createPatchSIFT() {
    auto sift = DescriptorFactory::create(DescriptorType::SIFT);
    // Keypoint size of 41 gives good coverage of 65x65 patch
    // (41/2 = 20.5 radius, SIFT uses ~4x radius = ~82px, scaled down)
    return std::make_unique<PatchTraditionalExtractor>(std::move(sift), 41.0f, false, DescriptorType::SIFT);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFT() {
    auto rgbsift = DescriptorFactory::create(DescriptorType::RGBSIFT);
    return std::make_unique<PatchTraditionalExtractor>(std::move(rgbsift), 41.0f, true, DescriptorType::RGBSIFT);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFTChannelAvg() {
    auto rgbsift_avg = DescriptorFactory::create(DescriptorType::RGBSIFT_CHANNEL_AVG);
    return std::make_unique<PatchTraditionalExtractor>(std::move(rgbsift_avg), 41.0f, true, DescriptorType::RGBSIFT_CHANNEL_AVG);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchHoNC() {
    auto honc = DescriptorFactory::create(DescriptorType::HoNC);
    return std::make_unique<PatchTraditionalExtractor>(std::move(honc), 41.0f, true, DescriptorType::HoNC);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchDSPSIFT() {
    auto dspsift = DescriptorFactory::create(DescriptorType::DSPSIFT_V2);
    return std::make_unique<PatchTraditionalExtractor>(std::move(dspsift), 41.0f, false, DescriptorType::DSPSIFT_V2);
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchSURF() {
    auto surf = DescriptorFactory::create(DescriptorType::SURF);
    return std::make_unique<PatchTraditionalExtractor>(std::move(surf), 41.0f, false, DescriptorType::SURF);
}

} // namespace patches
} // namespace thesis_project
