#pragma once

#include "thesis_project/types.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace thesis_project {

    /**
     * @brief Interface for keypoint detection algorithms
     * 
     * This interface provides a unified API for different keypoint detectors
     * (SIFT, Harris, ORB, etc.) with support for spatial constraints to
     * prevent descriptor window overlap.
     */
    class IKeypointGenerator {
    public:
        virtual ~IKeypointGenerator() = default;
        
        /**
         * @brief Detect keypoints in an image
         * @param image Input image (grayscale or color)
         * @param params Detection parameters
         * @return Vector of detected keypoints
         */
        virtual std::vector<cv::KeyPoint> detect(
            const cv::Mat& image,
            const KeypointParams& params = {}
        ) = 0;
        
        /**
         * @brief Detect keypoints with spatial non-overlap constraint
         * @param image Input image
         * @param min_distance Minimum euclidean distance between keypoint centers
         * @param params Detection parameters
         * @return Vector of non-overlapping keypoints
         */
        virtual std::vector<cv::KeyPoint> detectNonOverlapping(
            const cv::Mat& image,
            float min_distance,
            const KeypointParams& params = {}
        ) = 0;
        
        /**
         * @brief Get human-readable detector name
         * @return Detector name (e.g., "SIFT", "Harris", "ORB")
         */
        virtual std::string name() const = 0;
        
        /**
         * @brief Get detector type enum
         * @return KeypointGenerator enum value
         */
        virtual KeypointGenerator type() const = 0;
        
        /**
         * @brief Check if detector supports non-overlapping detection natively
         * @return true if detector has optimized non-overlapping implementation
         */
        virtual bool supportsNonOverlapping() const { return false; }
        
        /**
         * @brief Get recommended minimum distance for non-overlapping detection
         * @param descriptor_patch_size Expected descriptor patch size
         * @return Recommended minimum distance in pixels
         */
        virtual float getRecommendedMinDistance(int descriptor_patch_size = 32) const {
            return static_cast<float>(descriptor_patch_size);
        }
    };

} // namespace thesis_project