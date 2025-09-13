#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>

namespace thesis_project {

/**
 * @brief ORB (Oriented FAST and Rotated BRIEF) keypoint detector implementation
 * 
 * Wraps OpenCV's ORB detector with the IKeypointGenerator interface.
 * ORB provides fast keypoint detection suitable for real-time applications.
 */
class ORBKeypointGenerator : public IKeypointGenerator {
private:
    cv::Ptr<cv::ORB> detector_;
    
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
     * @brief Constructor with ORB detector parameters
     * @param num_features Maximum number of features to retain
     * @param scale_factor Pyramid decimation ratio
     * @param num_levels Number of pyramid levels
     * @param edge_threshold Border where features are not detected
     * @param first_level Level of pyramid to put source image to
     * @param wta_k Number of points that produce each element of BRIEF descriptor
     * @param score_type Algorithm used to rank features
     * @param patch_size Size of patch used by oriented BRIEF descriptor
     * @param fast_threshold FAST/AGAST detection threshold score
     */
    explicit ORBKeypointGenerator(
        int num_features = 500,
        float scale_factor = 1.2f,
        int num_levels = 8,
        int edge_threshold = 31,
        int first_level = 0,
        int wta_k = 2,
        cv::ORB::ScoreType score_type = cv::ORB::HARRIS_SCORE,
        int patch_size = 31,
        int fast_threshold = 20
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
    
    std::string name() const override { return "ORB"; }
    KeypointGenerator type() const override { return KeypointGenerator::ORB; }
    bool supportsNonOverlapping() const override { return true; }
    
    float getRecommendedMinDistance(int descriptor_patch_size = 32) const override {
        // ORB uses 31x31 patches by default, adjust accordingly
        return static_cast<float>(std::max(descriptor_patch_size, 31));
    }
};

} // namespace thesis_project