#pragma once

#include "MatchingStrategy.hpp"

namespace thesis_project::matching {

/**
 * @brief Brute-force matching strategy using OpenCV BFMatcher
 * 
 * This strategy implements straightforward brute-force matching using
 * OpenCV's BFMatcher with L2 norm and cross-check enabled.
 * 
 * Features:
 * - L2 norm distance metric (suitable for SIFT, SURF descriptors)
 * - Cross-check enabled for better match quality
 * - Simple threshold-based precision calculation
 * - Scale-adaptive threshold adjustment
 */
class BruteForceMatching : public MatchingStrategy {
public:
    /**
     * @brief Constructor with optional norm type
     * @param normType OpenCV norm type (default: NORM_L2)
     * @param crossCheck Whether to enable cross-check (default: true)
     */
    explicit BruteForceMatching(
        int normType = cv::NORM_L2, 
        bool crossCheck = true
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
        return "BruteForce";
    }
    
    bool supportsRatioTest() const override {
        return false; // Simple brute force doesn't use ratio test
    }

private:
    cv::BFMatcher matcher_;
};

} // namespace thesis_project::matching