#include "NonOverlappingKeypointGenerator.hpp"
#include <algorithm>
#include <cmath>

namespace thesis_project {

NonOverlappingKeypointGenerator::NonOverlappingKeypointGenerator(
    std::unique_ptr<IKeypointGenerator> base_detector,
    float default_min_distance
) : base_detector_(std::move(base_detector)),
    default_min_distance_(default_min_distance) {
    
    if (!base_detector_) {
        throw std::invalid_argument("Base detector cannot be null");
    }
    
    if (default_min_distance <= 0.0f) {
        throw std::invalid_argument("Default minimum distance must be positive");
    }
}

std::vector<cv::KeyPoint> NonOverlappingKeypointGenerator::detect(
    const cv::Mat& image,
    const KeypointParams& params
) {
    // Use detectNonOverlapping with default minimum distance
    return detectNonOverlapping(image, default_min_distance_, params);
}

std::vector<cv::KeyPoint> NonOverlappingKeypointGenerator::detectNonOverlapping(
    const cv::Mat& image,
    float min_distance,
    const KeypointParams& params
) {
    // Check if base detector supports non-overlapping natively
    if (base_detector_->supportsNonOverlapping()) {
        return base_detector_->detectNonOverlapping(image, min_distance, params);
    }
    
    // Otherwise, use base detection followed by filtering
    auto keypoints = base_detector_->detect(image, params);
    return filterOverlapping(std::move(keypoints), min_distance);
}

std::vector<cv::KeyPoint> NonOverlappingKeypointGenerator::filterOverlapping(
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