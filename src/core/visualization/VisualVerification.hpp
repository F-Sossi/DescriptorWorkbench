#ifndef CORE_VISUALIZATION_VISUAL_VERIFICATION_HPP
#define CORE_VISUALIZATION_VISUAL_VERIFICATION_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// Forward declaration to avoid circular dependency
struct experiment_config;

/**
 * @brief Handles visual verification and debugging displays for descriptor matching
 * 
 * This class consolidates all visualization functionality that was previously
 * scattered throughout the image processing pipeline.
 */
class VisualVerification {
public:
    /**
     * @brief Performs visual verification of descriptor matches between image pairs
     * @param folder Path to the image folder containing image sequence
     * @param results_folder Path to directory where results will be stored (currently unused)
     * @param config Experiment configuration specifying processing options
     */
    static void verifyMatches(const std::string& folder, 
                             const std::string& results_folder, 
                             const experiment_config& config);

    /**
     * @brief Performs visual verification of keypoint projection using homography matrices
     * @param folder Path to the image folder containing image sequence and homography files
     * @param results_folder Path to directory where results will be stored (currently unused)
     * @param config Experiment configuration specifying processing options
     */
    static void verifyHomography(const std::string& folder, 
                                const std::string& results_folder, 
                                const experiment_config& config);

private:
    /**
     * @brief Get color for keypoint visualization based on index
     * @param index Keypoint index for consistent coloring
     * @return OpenCV Scalar color value
     */
    static cv::Scalar getKeypointColor(size_t index);
};

#endif // CORE_VISUALIZATION_VISUAL_VERIFICATION_HPP