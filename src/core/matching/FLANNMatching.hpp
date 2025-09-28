#pragma once

#include "MatchingStrategy.hpp"
#include <opencv2/flann.hpp>

namespace thesis_project::matching {

/**
 * @brief FLANN-based matching strategy using OpenCV FlannBasedMatcher
 *
 * This strategy implements Fast Library for Approximate Nearest Neighbors (FLANN)
 * matching using OpenCV's FlannBasedMatcher with LSH (Locality Sensitive Hashing)
 * for binary descriptors and KDTree for float descriptors.
 *
 * Features:
 * - Automatic algorithm selection based on descriptor type
 * - LSH for binary descriptors (ORB, BRIEF, BRISK)
 * - KDTree for float descriptors (SIFT, SURF, VGG)
 * - Configurable search parameters for speed/accuracy trade-off
 * - Cross-check support for better match quality
 */
class FLANNMatching : public MatchingStrategy {
public:
    /**
     * @brief Constructor with configurable parameters
     * @param crossCheck Whether to enable cross-check (default: true)
     * @param trees Number of trees for KDTree (default: 5)
     * @param checks Number of checks for search (default: 50)
     */
    explicit FLANNMatching(
        bool crossCheck = true,
        int trees = 5,
        int checks = 50
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
        return "FLANN";
    }

    bool supportsRatioTest() const override {
        return true; // FLANN can be used with ratio test via knnMatch
    }

private:
    /**
     * @brief Determine if descriptors are binary based on type
     * @param descriptors Input descriptor matrix
     * @return bool True if descriptors are binary (CV_8U)
     */
    bool isBinaryDescriptor(const cv::Mat& descriptors) const;

    /**
     * @brief Create appropriate FLANN matcher for descriptor type
     * @param descriptors Sample descriptors to determine type
     * @return cv::Ptr<cv::FlannBasedMatcher> Configured matcher
     */
    cv::Ptr<cv::FlannBasedMatcher> createMatcher(const cv::Mat& descriptors) const;

    bool crossCheck_;
    int trees_;        // KDTree parameter
    int checks_;       // Search parameter
};

} // namespace thesis_project::matching