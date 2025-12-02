#include "helpers.hpp"
#include "src/core/experiment/ExperimentHelpers.hpp"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <fstream>

namespace experiment_helpers = thesis_project::experiment;

namespace thesis_project::cli::experiment_runner_helpers {

std::string normalizeDeviceString(const std::string& raw_device) {
    if (raw_device.empty()) {
        return "auto";
    }
    std::string normalized = raw_device;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "gpu" || normalized == "nvidia") {
        return "cuda";
    }
    if (normalized == "gpu+cpu" || normalized == "cpu+gpu" || normalized == "mixed") {
        return "mixed";
    }
    if (normalized == "cuda" || normalized == "cpu" || normalized == "auto") {
        return normalized;
    }
    return normalized;
}

cv::Mat generateMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                                   const std::vector<cv::KeyPoint>& kp1,
                                   const std::vector<cv::KeyPoint>& kp2,
                                   const std::vector<cv::DMatch>& matches,
                                   const std::vector<bool>& correctness) {
    cv::Mat visualization;

    cv::Mat color_img1, color_img2;
    if (img1.channels() == 1) {
        cv::cvtColor(img1, color_img1, cv::COLOR_GRAY2BGR);
    } else {
        color_img1 = img1.clone();
    }

    if (img2.channels() == 1) {
        cv::cvtColor(img2, color_img2, cv::COLOR_GRAY2BGR);
    } else {
        color_img2 = img2.clone();
    }

    std::vector<cv::Scalar> match_colors;
    match_colors.reserve(matches.size());

    for (size_t i = 0; i < matches.size(); ++i) {
        if (i < correctness.size() && correctness[i]) {
            match_colors.emplace_back(0, 255, 0);
        } else {
            match_colors.emplace_back(0, 0, 255);
        }
    }

    cv::drawMatches(color_img1, kp1, color_img2, kp2, matches, visualization,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (!visualization.empty()) {
        int img1_width = color_img1.cols;
        for (size_t i = 0; i < matches.size() && i < match_colors.size(); ++i) {
            const auto& match = matches[i];
            if (match.queryIdx < static_cast<int>(kp1.size()) &&
                match.trainIdx < static_cast<int>(kp2.size())) {

                cv::Point2f pt1 = kp1[match.queryIdx].pt;
                cv::Point2f pt2 = kp2[match.trainIdx].pt;
                pt2.x += img1_width;

                cv::line(visualization, pt1, pt2, match_colors[i], 2);
            }
        }
    }

    return visualization;
}

void maybeAccumulateTrueAveragePrecisionFromFile(
    const std::string& homographyPath,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors2,
    const std::string& sceneName,
    ::ExperimentMetrics& metrics) {

    std::ifstream hfile(homographyPath);
    if (!hfile.good()) {
        return;
    }

    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            hfile >> H.at<double>(r, c);
        }
    }

    if (H.empty()) {
        return;
    }

    experiment_helpers::accumulateTrueAveragePrecision(
        H, keypoints1, descriptors1, keypoints2, descriptors2, sceneName, metrics);
}

} // namespace thesis_project::cli::experiment_runner_helpers
