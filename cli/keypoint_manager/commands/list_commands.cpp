#include "list_commands.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include "thesis_project/logging.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace thesis_project::cli::keypoint_commands {

int listSets(thesis_project::database::DatabaseManager& db) {
    auto sets = db.getAvailableKeypointSets();
    std::cout << "Available keypoint sets (" << sets.size() << "):" << std::endl;
    for (const auto& [id, name, method] : sets) {
        std::cout << "ID " << id << ": " << name << " (" << method << ")" << std::endl;
    }
    if (sets.empty()) {
        std::cout << "  (No keypoint sets found - use generate-projected or generate-independent to create some)" << std::endl;
    }
    return 0;
}

int listScenes(thesis_project::database::DatabaseManager& db) {
    auto scenes = db.getAvailableScenes();
    std::cout << "📋 Available scenes (" << scenes.size() << "):" << std::endl;
    for (const auto& scene : scenes) {
        auto images = db.getAvailableImages(scene);
        int total_keypoints = 0;
        for (const auto& image : images) {
            auto keypoints = db.getLockedKeypoints(scene, image);
            total_keypoints += static_cast<int>(keypoints.size());
        }
        std::cout << "  📁 " << scene << " (" << images.size() << " images, " << total_keypoints << " total keypoints)" << std::endl;
    }
    return 0;
}

int countKeypoints(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " count <scene> <image>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " count i_dome 1.ppm" << std::endl;
        return 1;
    }

    std::string scene = argv[2];
    std::string image = argv[3];
    auto keypoints = db.getLockedKeypoints(scene, image);
    std::cout << "🔢 Keypoints for " << scene << "/" << image << ": " << keypoints.size() << std::endl;
    return 0;
}

int listDetectors() {
    auto detectors = KeypointGeneratorFactory::getSupportedDetectors();
    std::cout << "Supported detectors (" << detectors.size() << "):" << std::endl;
    for (const auto& detector : detectors) {
        float recommended_distance = KeypointGeneratorFactory::getRecommendedMinDistance(
            KeypointGeneratorFactory::parseDetectorType(detector), 32);
        std::cout << detector << " (recommended min_distance for 32px patches: "
                  << recommended_distance << "px)" << std::endl;
    }
    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
