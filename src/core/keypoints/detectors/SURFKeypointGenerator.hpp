#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>

namespace thesis_project {

/**
 * @brief SURF keypoint detector implementation
 *
 * Wraps OpenCV's SURF (Speeded-Up Robust Features) detector with the IKeypointGenerator interface.
 * SURF uses a Hessian-based blob detector which is faster than SIFT while maintaining
 * good performance. This allows fair comparison of SURF descriptors with SURF-native keypoints.
 */
class SURFKeypointGenerator : public IKeypointGenerator {
private:
    cv::Ptr<cv::xfeatures2d::SURF> detector_;

    /**
     * @brief Apply boundary filtering to keypoints
     * @param keypoints Input keypoints to filter
     * @param image_size Image dimensions for boundary checking
     * @param border_size Border size in pixels (default 40px)
     * @return Filtered keypoints
     */
    static std::vector<cv::KeyPoint> applyBoundaryFilter(
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Size& image_size,
        int border_size = 40
    );

    /**
     * @brief Apply response-based keypoint limiting
     * @param keypoints Input keypoints
     * @param max_keypoints Maximum number of keypoints to keep
     * @return Limited keypoints sorted by response strength
     */
    static std::vector<cv::KeyPoint> applyKeypointLimit(
        std::vector<cv::KeyPoint> keypoints,
        int max_keypoints
    );

    /**
     * @brief Filter overlapping keypoints using greedy selection
     * @param keypoints Input keypoints (will be sorted by response)
     * @param min_distance Minimum euclidean distance between keypoints
     * @return Non-overlapping keypoints
     */
    static std::vector<cv::KeyPoint> filterOverlapping(
        std::vector<cv::KeyPoint> keypoints,
        float min_distance
    );

public:
    /**
     * @brief Constructor with optional SURF parameters
     * @param hessian_threshold Hessian threshold for blob detection (default 400)
     * @param num_octaves Number of pyramid octaves (default 4)
     * @param num_octave_layers Number of layers within each octave (default 3)
     * @param extended Use extended 128-element descriptors (default false = 64 elements)
     * @param upright Don't compute orientation, faster (default false = compute orientation)
     */
    explicit SURFKeypointGenerator(
        double hessian_threshold = 400.0,
        int num_octaves = 4,
        int num_octave_layers = 3,
        bool extended = false,
        bool upright = false
    );

    // IKeypointGenerator interface implementation
    std::vector<cv::KeyPoint> detect(
        const cv::Mat& image,
        const KeypointParams& params
    ) override;

    std::vector<cv::KeyPoint> detectNonOverlapping(
        const cv::Mat& image,
        float min_distance,
        const KeypointParams& params
    ) override;

    [[nodiscard]] std::string name() const override { return "SURF"; }
    [[nodiscard]] KeypointGenerator type() const override { return KeypointGenerator::SURF; }
    [[nodiscard]] bool supportsNonOverlapping() const override { return true; }

    [[nodiscard]] float getRecommendedMinDistance(const int descriptor_patch_size = 32) const override {
        // For SURF, recommend patch size for no overlap
        // SURF typically uses ~20px patches
        return static_cast<float>(descriptor_patch_size);
    }
};

} // namespace thesis_project
