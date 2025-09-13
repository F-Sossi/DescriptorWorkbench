#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include <memory>

namespace thesis_project {

/**
 * @brief Decorator that adds non-overlapping constraint to any keypoint detector
 * 
 * This decorator wraps any IKeypointGenerator implementation and applies
 * spatial filtering to ensure minimum distance between detected keypoints.
 * Particularly useful for CNN descriptors where overlapping patches reduce
 * discriminative power.
 */
class NonOverlappingKeypointGenerator : public IKeypointGenerator {
private:
    std::unique_ptr<IKeypointGenerator> base_detector_;
    float default_min_distance_;
    
    /**
     * @brief Apply greedy spatial filtering to remove overlapping keypoints
     * @param keypoints Input keypoints (will be sorted by response)
     * @param min_distance Minimum euclidean distance between keypoints
     * @return Filtered keypoints with no overlaps
     */
    std::vector<cv::KeyPoint> filterOverlapping(
        std::vector<cv::KeyPoint> keypoints,
        float min_distance
    ) const;

public:
    /**
     * @brief Constructor
     * @param base_detector Base detector to wrap
     * @param default_min_distance Default minimum distance for non-overlapping detection
     */
    explicit NonOverlappingKeypointGenerator(
        std::unique_ptr<IKeypointGenerator> base_detector,
        float default_min_distance = 32.0f
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
    
    std::string name() const override { 
        return base_detector_->name() + "_NonOverlapping"; 
    }
    
    KeypointGenerator type() const override { 
        return base_detector_->type(); 
    }
    
    bool supportsNonOverlapping() const override { return true; }
    
    float getRecommendedMinDistance(int descriptor_patch_size = 32) const override {
        return base_detector_->getRecommendedMinDistance(descriptor_patch_size);
    }
    
    /**
     * @brief Get the underlying base detector
     * @return Pointer to the wrapped detector
     */
    const IKeypointGenerator* getBaseDetector() const {
        return base_detector_.get();
    }
    
    /**
     * @brief Get the default minimum distance
     * @return Default minimum distance in pixels
     */
    float getDefaultMinDistance() const {
        return default_min_distance_;
    }
};

} // namespace thesis_project