#include "BruteForceMatching.hpp"

namespace thesis_project::matching {

BruteForceMatching::BruteForceMatching(int normType, bool crossCheck) 
    : matcher_(normType, crossCheck) {
}

std::vector<cv::DMatch> BruteForceMatching::matchDescriptors(
    const cv::Mat& descriptors1, 
    const cv::Mat& descriptors2
) {
    std::vector<cv::DMatch> matches;
    matcher_.match(descriptors1, descriptors2, matches);
    return matches;
}

double BruteForceMatching::calculatePrecision(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::Point2f>& projectedPoints,
    double matchThreshold
) {
    int truePositives = 0;
    for (const auto& match : matches) {
        // Calculate the distance between the projected point and the corresponding match point in the second image
        if (cv::norm(projectedPoints[match.queryIdx] - keypoints2[match.trainIdx].pt) <= matchThreshold) {
            truePositives++;
        }
    }
    // Calculate precision
    return matches.empty() ? 0 : static_cast<double>(truePositives) / matches.size();
}

double BruteForceMatching::adjustMatchThreshold(
    double baseThreshold, 
    double scaleFactor
) {
    // Adjust the threshold based on the scale factor
    return baseThreshold * scaleFactor;
}

} // namespace thesis_project::matching