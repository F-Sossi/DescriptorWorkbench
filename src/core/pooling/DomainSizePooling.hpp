#pragma once

#include "PoolingStrategy.hpp"
#include "src/core/config/ExperimentConfig.hpp"

namespace thesis_project::pooling {

/**
 * @brief Domain Size Pooling (DSP) strategy with true pyramid-aware sampling
 *
 * Implements domain size pooling by building a SIFT-style Gaussian pyramid and
 * sampling descriptors from appropriate pyramid levels. This matches DSPSIFT's
 * approach of reusing the same scale-space structure for multiple scales.
 *
 * The algorithm (true pyramid version):
 * 1. Build a Gaussian pyramid once (like SIFT does internally)
 * 2. For each scale factor (e.g., 0.85x, 1.0x, 1.30x):
 *    - Find the appropriate pyramid level (octave, scale)
 *    - Scale keypoint sizes and positions for that level
 *    - Extract descriptors from the pre-built pyramid level
 * 3. Average the resulting descriptors with optional weighting
 * 4. Apply normalization if configured
 *
 * This ensures scale-space consistency by using the same pyramid SIFT would use,
 * avoiding the computational waste of rebuilding pyramids for each scale.
 */
class DomainSizePooling : public PoolingStrategy {
public:
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

    /**
     * @brief Build a SIFT-style Gaussian pyramid
     * @param image Input image (should be grayscale)
     * @param pyramid Output pyramid structure [octave][scale]
     * @param nOctaves Number of octaves to build (default 4)
     * @param nScalesPerOctave Scales per octave (default 3)
     * @param sigma0 Base sigma (default 1.6)
     */
    void buildGaussianPyramid(const cv::Mat& image,
                              std::vector<std::vector<cv::Mat>>& pyramid,
                              int nOctaves = 4,
                              int nScalesPerOctave = 3,
                              double sigma0 = 1.6) const;

    /**
     * @brief Find the appropriate pyramid level for a given scale factor
     * @param scaleFactor Scale factor (e.g., 0.85, 1.0, 1.30)
     * @param nScalesPerOctave Scales per octave
     * @return Pair of (octave_index, scale_index)
     */
    std::pair<int, int> findPyramidLevel(float scaleFactor, int nScalesPerOctave = 3) const;

    /**
     * @brief Aggregate descriptors from multiple scales using specified method
     * @param descriptors_per_scale Vector of descriptor matrices from each scale
     * @param weights Weights for each scale (for weighted average)
     * @param output Output aggregated descriptors
     * @param aggregation Aggregation method to use
     */
    void aggregateDescriptors(const std::vector<cv::Mat>& descriptors_per_scale,
                             const std::vector<double>& weights,
                             cv::Mat& output,
                             thesis_project::PoolingAggregation aggregation) const;
};

} // namespace thesis_project::pooling
