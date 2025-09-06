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
        const cv::Ptr<cv::Feature2D>& detector,
        const experiment_config& config
    ) override;

    // New-interface overload using IDescriptorExtractor wrappers
    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const experiment_config& config
    ) override;

    // Schema v1 overload (descriptor params only)
    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
    ) override;

    std::string getName() const override {
        return "Stacking";
    }

    float getDimensionalityMultiplier() const override {
        return 2.0f; // Combines two descriptors
    }

    bool requiresColorInput() const override {
        return true; // May need color conversion for secondary descriptor
    }

private:
    /**
     * @brief Prepare image for descriptor computation based on color space requirements
     * @param sourceImage Original input image
     * @param targetColorSpace Required color space (D_BW or D_COLOR)
     * @return cv::Mat Image in the appropriate color space
     */
    cv::Mat prepareImageForColorSpace(const cv::Mat& sourceImage, int targetColorSpace) const;
    
    /**
     * @brief Compute descriptors using appropriate detector interface
     * @param image Input image 
     * @param keypoints Keypoints to compute descriptors for
     * @param detector Feature detector/descriptor extractor
     * @return cv::Mat Computed descriptors
     */
    cv::Mat computeDescriptorsWithDetector(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        const cv::Ptr<cv::Feature2D>& detector
    ) const;
};

} // namespace thesis_project::pooling
