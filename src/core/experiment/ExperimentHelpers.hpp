#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/pooling/PoolingStrategy.hpp"
#include "src/core/matching/MatchingStrategy.hpp"
#include "src/core/config/ExperimentConfig.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace thesis_project::experiment {

/// Compute descriptors for a set of keypoints using the configured pooling strategy.
cv::Mat computeDescriptorsWithPooling(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    thesis_project::pooling::PoolingStrategy& poolingStrategy,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg);

/// Aggregated data returned from computeMatches.
struct MatchArtifacts {
    std::vector<cv::DMatch> matches;
    std::vector<bool> correctnessFlags;
    int correctMatches = 0;
};

/// Run descriptor matching and optionally evaluate correctness when a one-to-one mapping is available.
MatchArtifacts computeMatches(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    thesis_project::matching::MatchingStrategy& matcher,
    bool evaluateCorrectness,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2);

/// Accumulate true average precision metrics for a single image pair.
void accumulateTrueAveragePrecision(
    const cv::Mat& homography,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors2,
    const std::string& sceneName,
    ::ExperimentMetrics& metrics,
    double inlierThreshold = 3.0);

} // namespace thesis_project::experiment

