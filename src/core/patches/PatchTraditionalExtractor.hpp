#pragma once

#include "PatchDescriptorExtractor.hpp"
#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <string>

namespace thesis_project {
namespace patches {

/**
 * @brief Extracts traditional descriptors (SIFT, RGBSIFT, HoNC, etc.) from pre-extracted patches
 *
 * For traditional descriptors that expect full images + keypoints, this class:
 * 1. Creates a synthetic keypoint at the patch center
 * 2. Treats the 65x65 patch as a mini-image
 * 3. Delegates to the existing IDescriptorExtractor implementation
 */
class PatchTraditionalExtractor : public IPatchDescriptorExtractor {
public:
    /**
     * @brief Construct from an existing descriptor extractor
     * @param base_extractor The underlying descriptor extractor (SIFT, RGBSIFT, etc.)
     * @param keypoint_size Size of the synthetic keypoint (determines descriptor support region)
     * @param force_color Convert grayscale patches to 3-channel BGR before extraction
     */
    explicit PatchTraditionalExtractor(
        std::unique_ptr<IDescriptorExtractor> base_extractor,
        float keypoint_size = 41.0f,
        bool force_color = false,
        DescriptorType descriptor_type = DescriptorType::SIFT);

    ~PatchTraditionalExtractor() override = default;

    /**
     * @brief Extract descriptors from pre-extracted patches
     *
     * For each patch:
     * 1. Create a synthetic keypoint at center (32.5, 32.5) with the configured size
     * 2. Call the base extractor's extract() method treating the patch as an image
     * 3. Collect all descriptors
     *
     * @param patches Vector of 65x65 grayscale patches
     * @param params Descriptor parameters
     * @return cv::Mat where each row is a descriptor for the corresponding patch
     */
    cv::Mat extractFromPatches(
        const std::vector<cv::Mat>& patches,
        const DescriptorParams& params) override;

    std::string name() const override { return base_extractor_->name(); }
    int descriptorSize() const override { return base_extractor_->descriptorSize(); }
    int descriptorType() const override { return base_extractor_->descriptorType(); }

    // Traditional descriptors can work directly on 65x65 patches
    bool requiresResize() const override { return false; }
    int expectedPatchSize() const override { return 65; }
    std::unique_ptr<IPatchDescriptorExtractor> clone() const override;

private:
    std::unique_ptr<IDescriptorExtractor> base_extractor_;
    float keypoint_size_;
    bool force_color_;
    DescriptorType descriptor_type_;
};

/**
 * @brief Factory functions for common traditional descriptors on patches
 */
std::unique_ptr<IPatchDescriptorExtractor> createPatchSIFT();
std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFT();
std::unique_ptr<IPatchDescriptorExtractor> createPatchRGBSIFTChannelAvg();
std::unique_ptr<IPatchDescriptorExtractor> createPatchHoNC();
std::unique_ptr<IPatchDescriptorExtractor> createPatchDSPSIFT();
std::unique_ptr<IPatchDescriptorExtractor> createPatchSURF();

} // namespace patches
} // namespace thesis_project
