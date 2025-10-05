#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "src/core/config/ExperimentConfig.hpp"

// Forward declare new interface extractor to avoid heavy includes
namespace thesis_project { class IDescriptorExtractor; }

namespace thesis_project::pooling {

/**
 * @brief Abstract base class for all pooling strategies
 * 
 * This interface defines how descriptor modifications like Domain Size Pooling
 * and Stacking should be implemented. Each pooling strategy takes an image,
 * keypoints, and detector, then returns modified descriptors.
 */
class PoolingStrategy {
public:
    virtual ~PoolingStrategy() = default;

    /**
     * @brief Apply the pooling strategy to compute descriptors
     * 
     * @param image Input image (color or grayscale depending on descriptor)
     * @param keypoints Detected keypoints to compute descriptors for
     * @param detector Primary feature detector/descriptor extractor
     * @param config Experiment configuration containing pooling parameters
     * @return cv::Mat Computed descriptors after applying pooling strategy
     */
    virtual cv::Mat computeDescriptors(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        thesis_project::IDescriptorExtractor& extractor,
        const thesis_project::config::ExperimentConfig::DescriptorConfig& /*descCfg*/
    )
    {
        throw std::runtime_error("PoolingStrategy v1 path not implemented for this strategy");
    }

    /**
     * @brief Get human-readable name of the pooling strategy
     * @return std::string Strategy name for logging and identification
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Get expected descriptor dimensionality multiplier
     * 
     * For example:
     * - NoPooling: returns 1.0 (same as base descriptor)
     * - Stacking: returns 2.0 (combines two descriptors)
     * - DSP: returns 1.0 (same dimension, different computation)
     * 
     * @return float Expected dimensionality multiplier
     */
    virtual float getDimensionalityMultiplier() const = 0;

    /**
     * @brief Check if this strategy requires color input
     * @return bool True if color image required, false if grayscale is fine
     */
    virtual bool requiresColorInput() const = 0;
};

// Type alias for shared pointer to pooling strategy
using PoolingStrategyPtr = std::unique_ptr<PoolingStrategy>;

} // namespace thesis_project::pooling
