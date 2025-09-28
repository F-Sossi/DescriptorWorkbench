#include "FLANNMatching.hpp"
#include <opencv2/flann.hpp>

namespace thesis_project::matching {

FLANNMatching::FLANNMatching(bool crossCheck, int trees, int checks)
    : crossCheck_(crossCheck), trees_(trees), checks_(checks) {
}

std::vector<cv::DMatch> FLANNMatching::matchDescriptors(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2
) {
    std::vector<cv::DMatch> matches;

    if (descriptors1.empty() || descriptors2.empty()) {
        return matches;
    }

    // Create appropriate matcher based on descriptor type
    auto matcher = createMatcher(descriptors1);

    if (crossCheck_) {
        // Perform cross-check matching for better quality
        std::vector<cv::DMatch> matches12, matches21;

        matcher->match(descriptors1, descriptors2, matches12);
        matcher->match(descriptors2, descriptors1, matches21);

        // Cross-check: keep only matches that are mutual
        for (const auto& match12 : matches12) {
            for (const auto& match21 : matches21) {
                if (match12.queryIdx == match21.trainIdx &&
                    match12.trainIdx == match21.queryIdx) {
                    matches.push_back(match12);
                    break;
                }
            }
        }
    } else {
        // Simple one-way matching
        matcher->match(descriptors1, descriptors2, matches);
    }

    return matches;
}

double FLANNMatching::calculatePrecision(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::Point2f>& projectedPoints,
    double matchThreshold
) {
    // Use same precision calculation as BruteForce for consistency
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

double FLANNMatching::adjustMatchThreshold(
    double baseThreshold,
    double scaleFactor
) {
    // Adjust the threshold based on the scale factor (same as BruteForce)
    return baseThreshold * scaleFactor;
}

bool FLANNMatching::isBinaryDescriptor(const cv::Mat& descriptors) const {
    // Binary descriptors are typically CV_8U (8-bit unsigned)
    // Float descriptors are typically CV_32F (32-bit float)
    return descriptors.type() == CV_8U || descriptors.type() == CV_8UC1;
}

cv::Ptr<cv::FlannBasedMatcher> FLANNMatching::createMatcher(const cv::Mat& descriptors) const {
    if (isBinaryDescriptor(descriptors)) {
        // Use LSH (Locality Sensitive Hashing) for binary descriptors
        auto indexParams = cv::makePtr<cv::flann::LshIndexParams>(
            6,     // table_number: number of hash tables
            12,    // key_size: length of the key in the hash tables
            1      // multi_probe_level: level of multiprobe
        );
        auto searchParams = cv::makePtr<cv::flann::SearchParams>(checks_);
        return cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
    } else {
        // Use KDTree for float descriptors (SIFT, SURF, VGG, etc.)
        auto indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(trees_);
        auto searchParams = cv::makePtr<cv::flann::SearchParams>(checks_);
        return cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
    }
}

} // namespace thesis_project::matching