#pragma once

#include "MatchingStrategy.hpp"

namespace thesis_project::matching {

/**
 * @brief SNN (Second Nearest Neighbor) Ratio Test matching strategy
 *
 * This strategy implements Lowe's ratio test for robust keypoint matching.
 * The test filters matches by comparing the distance to the best match against
 * the distance to the second-best match.
 *
 * Algorithm:
 * 1. For each descriptor, find the 2 nearest neighbors
 * 2. Calculate ratio = distance_to_1st / distance_to_2nd
 * 3. Accept match only if ratio < threshold (default 0.8)
 *
 * Features:
 * - Eliminates ~90% of false matches while keeping ~95% of correct matches
 * - Works with any descriptor type (SIFT, HardNet, SOSNet, etc.)
 * - Standard evaluation method used in literature
 * - Configurable ratio threshold
 *
 * References:
 * - Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
 * - Brown PhotoTour Revisited benchmark uses this for all evaluations
 */
class RatioTestMatching : public MatchingStrategy {
public:
    /**
     * @brief Constructor with configurable ratio threshold
     * @param ratioThreshold Ratio threshold for accepting matches (default: 0.8)
     * @param normType OpenCV norm type (default: NORM_L2)
     */
    explicit RatioTestMatching(
        float ratioThreshold = 0.8f,
        int normType = cv::NORM_L2
    );

    std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2
    ) override;

    double calculatePrecision(
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::Point2f>& projectedPoints,
        double matchThreshold
    ) override;

    double adjustMatchThreshold(
        double baseThreshold,
        double scaleFactor
    ) override;

    std::string getName() const override {
        return "RatioTest";
    }

    bool supportsRatioTest() const override {
        return true; // This IS the ratio test
    }

    /**
     * @brief Get the current ratio threshold
     * @return float Current ratio threshold value
     */
    float getRatioThreshold() const {
        return ratioThreshold_;
    }

    /**
     * @brief Set a new ratio threshold
     * @param threshold New ratio threshold (should be < 1.0)
     */
    void setRatioThreshold(float threshold) {
        ratioThreshold_ = threshold;
    }

private:
    float ratioThreshold_;  ///< Ratio threshold for accepting matches
    cv::BFMatcher matcher_; ///< OpenCV matcher for kNN search
};

} // namespace thesis_project::matching