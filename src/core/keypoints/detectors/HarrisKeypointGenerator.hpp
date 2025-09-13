#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>

namespace thesis_project {

/**
 * @brief Harris corner detector implementation
 * 
 * Wraps OpenCV's goodFeaturesToTrack function with Harris corner detection
 * parameters. Provides both standard detection and non-overlapping capabilities.
 */
class HarrisKeypointGenerator : public IKeypointGenerator {
private:
    // Harris detector parameters
    int max_corners_;
    double quality_level_;
    double min_distance_;
    int block_size_;
    bool use_harris_detector_;
    double k_;
    
    /**
     * @brief Apply boundary filtering to keypoints
     */
    std::vector<cv::KeyPoint> applyBoundaryFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Size& image_size,
        int border_size = 40
    ) const;
    
    /**
     * @brief Apply keypoint limiting based on response strength
     */
    std::vector<cv::KeyPoint> applyKeypointLimit(
        std::vector<cv::KeyPoint> keypoints,
        int max_keypoints
    ) const;
    
    /**
     * @brief Filter overlapping keypoints
     */
    std::vector<cv::KeyPoint> filterOverlapping(
        std::vector<cv::KeyPoint> keypoints,
        float min_distance
    ) const;

public:
    /**
     * @brief Constructor with Harris detector parameters
     * @param max_corners Maximum number of corners to detect
     * @param quality_level Quality level for corner detection
     * @param min_distance Minimum distance between corners
     * @param block_size Block size for Harris detector
     * @param use_harris_detector Use Harris detector (true) or eigenvalues (false)
     * @param k Harris detector free parameter
     */
    explicit HarrisKeypointGenerator(
        int max_corners = 1000,
        double quality_level = 0.01,
        double min_distance = 10.0,
        int block_size = 3,
        bool use_harris_detector = true,
        double k = 0.04
    );
    
    // IKeypointGenerator interface implementation
    std::vector<cv::KeyPoint> detect(
        const cv::Mat& image,
        const KeypointParams& params = {}
    ) override;
    
    std::vector<cv::KeyPoint> detectNonOverlapping(
        const cv::Mat& image,
        float min_distance,
        const KeypointParams& params = {}
    ) override;
    
    std::string name() const override { return "Harris"; }
    KeypointGenerator type() const override { return KeypointGenerator::HARRIS; }
    bool supportsNonOverlapping() const override { return true; }
    
    float getRecommendedMinDistance(int descriptor_patch_size = 32) const override {
        // Harris corners are typically more sparse, can use smaller distance
        return static_cast<float>(descriptor_patch_size * 0.8f);
    }
};

} // namespace thesis_project