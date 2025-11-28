#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

#include "src/core/metrics/ExperimentMetrics.hpp"
#include <map>

namespace thesis_project::cli::experiment_runner_helpers {

// Normalize device strings to canonical forms (cpu/cuda/auto/mixed).
std::string normalizeDeviceString(const std::string& raw_device);

// Build a color-coded match visualization, highlighting correct vs incorrect matches.
cv::Mat generateMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                                   const std::vector<cv::KeyPoint>& kp1,
                                   const std::vector<cv::KeyPoint>& kp2,
                                   const std::vector<cv::DMatch>& matches,
                                   const std::vector<bool>& correctness);

// Load homography (if present) and accumulate true average precision metrics.
void maybeAccumulateTrueAveragePrecisionFromFile(
    const std::string& homographyPath,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors2,
    const std::string& sceneName,
    ::ExperimentMetrics& metrics);

template <typename BufferT>
void registerSceneHomography(BufferT* buffer,
                             const std::string& from_image,
                             const std::string& to_image,
                             const cv::Mat& homography,
                             const std::string& scene_name) {
    if (!buffer || homography.empty()) return;
    buffer->scene_name = scene_name;
    std::string key = from_image + "_" + to_image;
    buffer->homographies[key] = homography;
}

} // namespace thesis_project::cli::experiment_runner_helpers
