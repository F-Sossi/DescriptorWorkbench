#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include "thesis_project/types.hpp"
#include <memory>
#include <string>

namespace thesis_project {

/**
 * @brief Factory for creating keypoint generators
 * 
 * Provides a unified interface for creating different types of keypoint
 * detectors with optional non-overlapping constraints. Supports all
 * detectors defined in the KeypointGenerator enum.
 */
class KeypointGeneratorFactory {
public:
    /**
     * @brief Create a keypoint generator of the specified type
     * @param type Type of detector to create
     * @param non_overlapping Whether to apply non-overlapping constraint
     * @param min_distance Minimum distance for non-overlapping (used if non_overlapping=true)
     * @return Unique pointer to the created detector
     * @throws std::invalid_argument if type is unsupported
     */
    static std::unique_ptr<IKeypointGenerator> create(
        KeypointGenerator type,
        bool non_overlapping = false,
        float min_distance = 32.0f
    );
    
    /**
     * @brief Create a keypoint generator from string name
     * @param detector_name Name of detector ("sift", "harris", "orb", "keynet")
     * @param non_overlapping Whether to apply non-overlapping constraint
     * @param min_distance Minimum distance for non-overlapping
     * @return Unique pointer to the created detector
     * @throws std::invalid_argument if detector_name is unknown
     */
    static std::unique_ptr<IKeypointGenerator> createFromString(
        const std::string& detector_name,
        bool non_overlapping = false,
        float min_distance = 32.0f
    );
    
    /**
     * @brief Create a SIFT detector with default parameters
     * @return Unique pointer to SIFT detector
     */
    static std::unique_ptr<IKeypointGenerator> createSIFT();
    
    /**
     * @brief Create a Harris detector with default parameters
     * @return Unique pointer to Harris detector
     */
    static std::unique_ptr<IKeypointGenerator> createHarris();
    
    /**
     * @brief Create an ORB detector with default parameters
     * @return Unique pointer to ORB detector
     */
    static std::unique_ptr<IKeypointGenerator> createORB();

    /**
     * @brief Create a KeyNet detector with default parameters
     * @return Unique pointer to KeyNet detector
     */
    static std::unique_ptr<IKeypointGenerator> createKeyNet();

    /**
     * @brief Wrap any detector with non-overlapping constraint
     * @param base_detector Base detector to wrap
     * @param min_distance Minimum distance between keypoints
     * @return Unique pointer to wrapped detector
     * @throws std::invalid_argument if base_detector is null
     */
    static std::unique_ptr<IKeypointGenerator> makeNonOverlapping(
        std::unique_ptr<IKeypointGenerator> base_detector,
        float min_distance = 32.0f
    );
    
    /**
     * @brief Parse detector type from string
     * @param detector_str String representation ("sift", "harris", "orb", "keynet")
     * @return KeypointGenerator enum value
     * @throws std::invalid_argument if string is not recognized
     */
    static KeypointGenerator parseDetectorType(const std::string& detector_str);
    
    /**
     * @brief Get list of supported detector names
     * @return Vector of supported detector name strings
     */
    static std::vector<std::string> getSupportedDetectors();
    
    /**
     * @brief Check if a detector type is supported
     * @param type Detector type to check
     * @return true if supported, false otherwise
     */
    static bool isSupported(KeypointGenerator type);
    
    /**
     * @brief Get recommended minimum distance for a detector and patch size
     * @param type Detector type
     * @param descriptor_patch_size Expected descriptor patch size
     * @return Recommended minimum distance in pixels
     */
    static float getRecommendedMinDistance(
        KeypointGenerator type,
        int descriptor_patch_size = 32
    );
};

} // namespace thesis_project