#pragma once

#include "thesis_project/types.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include <optional>

namespace thesis_project::keypoints {

    struct HydrationStats {
        size_t total = 0;
        size_t hydrated = 0;
    };

    /**
     * @brief Map descriptor type to preferred detector for attribute hydration.
     * @return Detector enum if a preferred mapping exists.
     */
    std::optional<KeypointGenerator> preferredDetectorForDescriptor(DescriptorType descriptor_type);

    /**
     * @brief Load keypoints from database and hydrate with detector-specific attributes when available.
     *
     * When detector attributes exist for the provided detector, the keypoint properties (size, angle,
     * response, octave, class_id) are replaced with the stored values. Otherwise the original
     * keypoints from the master lattice are returned unchanged.
     */
    HydrationStats loadKeypointsWithAttributes(
        database::DatabaseManager& db,
        int keypoint_set_id,
        const std::string& scene_name,
        const std::string& image_name,
        KeypointGenerator detector,
        std::vector<cv::KeyPoint>& out_keypoints);

} // namespace thesis_project::keypoints

