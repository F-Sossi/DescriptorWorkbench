#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>

namespace thesis_project {

/**
 * @brief SIFT keypoint detector implementation
 * 
 * Wraps OpenCV's SIFT detector with the IKeypointGenerator interface.
 * Provides both standard detection and non-overlapping detection capabilities.
 */
class SIFTKeypointGenerator : public IKeypointGenerator {
private:
    cv::Ptr<cv::SIFT> detector_;
    
    /**
     * @brief Apply boundary filtering to keypoints
     * @param keypoints Input keypoints to filter
     * @param image_size Image dimensions for boundary checking
     * @param border_size Border size in pixels (default 40px)
     * @return Filtered keypoints
     */
    std::vector<cv::KeyPoint> applyBoundaryFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Size& image_size,
        int border_size = 40
    ) const;
    
    /**
     * @brief Apply response-based keypoint limiting
     * @param keypoints Input keypoints
     * @param max_keypoints Maximum number of keypoints to keep
     * @return Limited keypoints sorted by response strength
     */
    std::vector<cv::KeyPoint> applyKeypointLimit(
        std::vector<cv::KeyPoint> keypoints,
        int max_keypoints
    ) const;
    
    /**
     * @brief Filter overlapping keypoints using greedy selection
     * @param keypoints Input keypoints (will be sorted by response)
     * @param min_distance Minimum euclidean distance between keypoints
     * @return Non-overlapping keypoints
     */
    std::vector<cv::KeyPoint> filterOverlapping(
        std::vector<cv::KeyPoint> keypoints,
        float min_distance
    ) const;

public:
    /**
     * @brief Constructor with optional SIFT parameters
     * @param num_features Maximum number of features (0 = unlimited)
     * @param num_octave_layers Number of layers in each octave
     * @param contrast_threshold Contrast threshold for feature detection
     * @param edge_threshold Edge threshold
     * @param sigma Initial sigma for Gaussian blur
     */
    explicit SIFTKeypointGenerator(
        int num_features = 0,
        int num_octave_layers = 3,
        double contrast_threshold = 0.04,
        double edge_threshold = 10,
        double sigma = 1.6
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
    
    std::string name() const override { return "SIFT"; }
    KeypointGenerator type() const override { return KeypointGenerator::SIFT; }
    bool supportsNonOverlapping() const override { return true; }
    
    float getRecommendedMinDistance(int descriptor_patch_size = 32) const override {
        // For SIFT, recommend patch size for no overlap
        return static_cast<float>(descriptor_patch_size);
    }
};

} // namespace thesis_project