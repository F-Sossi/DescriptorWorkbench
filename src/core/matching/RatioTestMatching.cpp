#include "RatioTestMatching.hpp"
#include <algorithm>
#include <iostream>

namespace thesis_project::matching {

RatioTestMatching::RatioTestMatching(float ratioThreshold, int normType)
    : ratioThreshold_(ratioThreshold), matcher_(normType, false) {
    // Note: crossCheck is disabled for ratio test since we do kNN matching
    if (ratioThreshold <= 0.0f || ratioThreshold >= 1.0f) {
        std::cerr << "Warning: Ratio threshold should be between 0 and 1, got: "
                  << ratioThreshold << ". Using default 0.8." << std::endl;
        ratioThreshold_ = 0.8f;
    }
}

std::vector<cv::DMatch> RatioTestMatching::matchDescriptors(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2
) {
    if (descriptors1.empty() || descriptors2.empty()) {
        return {};
    }

    // Step 1: Find 2 nearest neighbors for each query descriptor
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher_.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Step 2: Apply ratio test
    std::vector<cv::DMatch> goodMatches;
    goodMatches.reserve(knnMatches.size() / 2); // Rough estimate

    for (const auto& matchPair : knnMatches) {
        // We need exactly 2 matches to perform the ratio test
        if (matchPair.size() == 2) {
            const cv::DMatch& firstMatch = matchPair[0];   // Best match
            const cv::DMatch& secondMatch = matchPair[1];  // Second best match

            // Calculate ratio of distances
            float ratio = firstMatch.distance / secondMatch.distance;

            // Accept match if ratio is below threshold
            if (ratio < ratioThreshold_) {
                goodMatches.push_back(firstMatch);
            }
        }
        // If we only have 1 match (no second neighbor), we can't apply ratio test
        // In this case, we could either accept or reject - literature typically rejects
    }

    return goodMatches;
}

double RatioTestMatching::calculatePrecision(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::Point2f>& projectedPoints,
    double matchThreshold
) {
    if (matches.empty()) {
        return 0.0;
    }

    int truePositives = 0;
    for (const auto& match : matches) {
        // Ensure indices are valid
        if (match.queryIdx >= static_cast<int>(projectedPoints.size()) ||
            match.trainIdx >= static_cast<int>(keypoints2.size())) {
            continue;
        }

        // Calculate distance between projected point and matched keypoint
        const cv::Point2f& projectedPoint = projectedPoints[match.queryIdx];
        const cv::Point2f& matchedPoint = keypoints2[match.trainIdx].pt;

        double distance = cv::norm(projectedPoint - matchedPoint);

        if (distance <= matchThreshold) {
            truePositives++;
        }
    }

    return static_cast<double>(truePositives) / matches.size();
}

double RatioTestMatching::adjustMatchThreshold(
    double baseThreshold,
    double scaleFactor
) {
    // Same threshold adjustment as brute force - scale with image scale
    return baseThreshold * scaleFactor;
}

} // namespace thesis_project::matching