#include "SIFTKeypointGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace thesis_project {

SIFTKeypointGenerator::SIFTKeypointGenerator(
    int num_features,
    int num_octave_layers,
    double contrast_threshold,
    double edge_threshold,
    double sigma
) {
    detector_ = cv::SIFT::create(
        num_features,
        num_octave_layers,
        contrast_threshold,
        edge_threshold,
        sigma
    );
}

std::vector<cv::KeyPoint> SIFTKeypointGenerator::detect(
    const cv::Mat& image,
    const KeypointParams& params
) {
    if (image.empty()) {
        return {};
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }
    
    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(gray_image, keypoints);
    
    // Apply boundary filtering (40px border like existing implementation)
    keypoints = applyBoundaryFilter(keypoints, gray_image.size(), 40);
    
    // Apply keypoint limit from params
    if (params.max_features > 0 && keypoints.size() > static_cast<size_t>(params.max_features)) {
        keypoints = applyKeypointLimit(std::move(keypoints), params.max_features);
    }
    
    return keypoints;
}

std::vector<cv::KeyPoint> SIFTKeypointGenerator::detectNonOverlapping(
    const cv::Mat& image,
    float min_distance,
    const KeypointParams& params
) {
    // First get all keypoints using standard detection
    auto keypoints = detect(image, params);
    
    // Apply non-overlapping filter
    return filterOverlapping(std::move(keypoints), min_distance);
}

std::vector<cv::KeyPoint> SIFTKeypointGenerator::applyBoundaryFilter(
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Size& image_size,
    int border_size
) const {
    std::vector<cv::KeyPoint> filtered;
    filtered.reserve(keypoints.size());
    
    for (const auto& kp : keypoints) {
        if (kp.pt.x >= border_size && 
            kp.pt.y >= border_size &&
            kp.pt.x < (image_size.width - border_size) &&
            kp.pt.y < (image_size.height - border_size)) {
            filtered.push_back(kp);
        }
    }
    
    return filtered;
}

std::vector<cv::KeyPoint> SIFTKeypointGenerator::applyKeypointLimit(
    std::vector<cv::KeyPoint> keypoints,
    int max_keypoints
) const {
    if (keypoints.size() <= static_cast<size_t>(max_keypoints)) {
        return keypoints;
    }
    
    // Sort by response strength (descending)
    std::sort(keypoints.begin(), keypoints.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });
    
    // Keep only the top max_keypoints
    keypoints.resize(max_keypoints);
    return keypoints;
}

std::vector<cv::KeyPoint> SIFTKeypointGenerator::filterOverlapping(
    std::vector<cv::KeyPoint> keypoints,
    float min_distance
) const {
    if (keypoints.empty() || min_distance <= 0.0f) {
        return keypoints;
    }
    
    // Sort by response strength (best keypoints first)
    std::sort(keypoints.begin(), keypoints.end(),
        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
        });
    
    std::vector<cv::KeyPoint> filtered;
    filtered.reserve(keypoints.size());
    
    // Greedy selection: keep keypoint if it's far enough from all selected ones
    for (const auto& candidate : keypoints) {
        bool valid = true;
        
        for (const auto& selected : filtered) {
            float dx = candidate.pt.x - selected.pt.x;
            float dy = candidate.pt.y - selected.pt.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            
            if (distance < min_distance) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            filtered.push_back(candidate);
        }
    }
    
    return filtered;
}

} // namespace thesis_project