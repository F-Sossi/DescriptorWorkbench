#include "ExperimentHelpers.hpp"

#include "src/core/metrics/TrueAveragePrecision.hpp"
#include <algorithm>
#include <limits>

namespace thesis_project::experiment {

cv::Mat computeDescriptorsWithPooling(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    thesis_project::pooling::PoolingStrategy& poolingStrategy,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg)
{
    if (keypoints.empty()) {
        return cv::Mat();
    }

    try {
        return poolingStrategy.computeDescriptors(image, keypoints, extractor, descCfg);
    } catch (const std::exception&) {
        return cv::Mat();
    }
}

MatchArtifacts computeMatches(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    thesis_project::matching::MatchingStrategy& matcher,
    bool evaluateCorrectness,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2)
{
    MatchArtifacts artifacts;
    if (descriptors1.empty() || descriptors2.empty()) {
        return artifacts;
    }

    artifacts.matches = matcher.matchDescriptors(descriptors1, descriptors2);

    const size_t matchCount = artifacts.matches.size();
    artifacts.correctnessFlags.assign(matchCount, false);

    if (!evaluateCorrectness) {
        return artifacts;
    }

    for (size_t i = 0; i < matchCount; ++i) {
        const auto& match = artifacts.matches[i];
        bool isCorrect = match.queryIdx == match.trainIdx;
        artifacts.correctnessFlags[i] = isCorrect;
        if (isCorrect) {
            ++artifacts.correctMatches;
        }
    }

    return artifacts;
}

void accumulateTrueAveragePrecision(
    const cv::Mat& homography,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors2,
    const std::string& sceneName,
    ::ExperimentMetrics& metrics,
    double inlierThreshold)
{
    if (homography.empty() || descriptors1.empty() || descriptors2.empty()) {
        return;
    }

    const int queryCount = static_cast<int>(keypoints1.size());
    const int targetCount = static_cast<int>(keypoints2.size());
    if (queryCount == 0 || targetCount == 0) {
        return;
    }

    for (int q = 0; q < queryCount; ++q) {
        cv::Mat queryDescriptor = descriptors1.row(q);
        if (queryDescriptor.empty() || cv::norm(queryDescriptor) == 0.0) {
            auto emptyResult = TrueAveragePrecision::QueryAPResult{};
            emptyResult.has_potential_match = false;
            emptyResult.ap = 0.0;
            metrics.addQueryAP(sceneName, emptyResult);
            continue;
        }

        std::vector<double> distances;
        distances.reserve(targetCount);
        for (int t = 0; t < targetCount; ++t) {
            cv::Mat targetDescriptor = descriptors2.row(t);
            if (targetDescriptor.empty()) {
                distances.push_back(std::numeric_limits<double>::infinity());
                continue;
            }
            double distance = cv::norm(queryDescriptor, targetDescriptor, cv::NORM_L2SQR);
            distances.push_back(distance);
        }

        auto ap = TrueAveragePrecision::computeQueryAP(
            keypoints1[q], homography, keypoints2, distances, inlierThreshold);
        metrics.addQueryAP(sceneName, ap);
    }
}

} // namespace thesis_project::experiment

