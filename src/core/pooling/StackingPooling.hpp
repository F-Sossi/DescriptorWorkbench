#pragma once

#include "PoolingStrategy.hpp"
#include "src/core/config/ExperimentConfig.hpp"

namespace thesis_project::pooling {

/**
 * @brief Stacking pooling strategy
 * 
 * Implements descriptor stacking by computing two different descriptor types
 * on the same keypoints and concatenating them horizontally. This allows
 * combining complementary information from different descriptors.
 * 
 * For example:
 * - SIFT (grayscale) + RGBSIFT (color) = 128D + 384D = 512D descriptor
 * - SIFT + HoNC = grayscale + color histogram information
 * 
 * The algorithm:
 * 1. Prepare image for first descriptor (handle color space)
 * 2. Compute first descriptor using primary detector
 * 3. Prepare image for second descriptor (handle color space conversion)
 * 4. Compute second descriptor using secondary detector  
 * 5. Horizontally concatenate the descriptors
 */
class StackingPooling : public PoolingStrategy {
public:
    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
    ) override;

    std::string getName() const override { return "Stacking"; }
    float getDimensionalityMultiplier() const override { return 2.0f; }
    bool requiresColorInput() const override { return true; }

private:
    cv::Mat prepareImageForColorSpace(const cv::Mat& sourceImage, bool useColor) const;
    cv::Mat computeSecondaryDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::DescriptorType descriptorType,
        bool useColor,
        const thesis_project::DescriptorParams& params
    ) const;
};

} // namespace thesis_project::pooling
