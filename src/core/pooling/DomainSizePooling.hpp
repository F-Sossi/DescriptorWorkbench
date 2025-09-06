#pragma once

#include "PoolingStrategy.hpp"
#include "src/core/config/ExperimentConfig.hpp"

namespace thesis_project::pooling {

/**
 * @brief Domain Size Pooling (DSP) strategy
 * 
 * Implements domain size pooling by computing descriptors at multiple scales
 * and averaging them. This technique improves robustness to scale variations
 * in the image.
 * 
 * The algorithm:
 * 1. Resize image to multiple scales (defined in config.scales)
 * 2. Compute descriptors at each scale 
 * 3. Average the resulting descriptors
 * 4. Apply normalization if configured
 */
class DomainSizePooling : public PoolingStrategy {
public:
    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Ptr<cv::Feature2D>& detector,
        const experiment_config& config
    ) override;

    // New interface overload: pool using modern extractor API
    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const experiment_config& config
    ) override;

    cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
    ) override;

    std::string getName() const override {
        return "DomainSizePooling";
    }

    float getDimensionalityMultiplier() const override {
        return 1.0f; // Same dimensionality as base descriptor
    }

    bool requiresColorInput() const override {
        return false; // Works with any image type
    }

private:
    /**
     * @brief Apply rooting operation to descriptors if configured
     * @param descriptors Input/output descriptors to modify
     */
    void applyRooting(cv::Mat& descriptors) const;
};

} // namespace thesis_project::pooling
