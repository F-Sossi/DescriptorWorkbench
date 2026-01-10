#include "non_overlapping.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include "thesis_project/logging.hpp"
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace thesis_project::cli::keypoint_commands {

int generateNonOverlapping(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0] << " generate-non-overlapping <data_folder> <detector> <min_distance> [name]" << std::endl;
        std::cerr << "  Detectors: sift, harris, orb" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-non-overlapping ../data sift 32.0 cnn_optimized_keypoints" << std::endl;
        return 1;
    }

    std::string data_folder = argv[2];
    std::string detector_str = argv[3];
    float min_distance = std::stof(argv[4]);
    std::string set_name = (argc == 6) ? argv[5] : (detector_str + "_non_overlapping_" + std::to_string(std::time(nullptr)));

    try {
        KeypointGenerator detector_type = KeypointGeneratorFactory::parseDetectorType(detector_str);

        LOG_INFO("Generating non-overlapping keypoints using " + detector_str + " detector");
        LOG_INFO("Data folder: " + data_folder);
        LOG_INFO("Minimum distance: " + std::to_string(min_distance) + "px");
        LOG_INFO("Keypoint set name: " + set_name);

        namespace fs = boost::filesystem;
        if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
            std::cerr << "Data folder does not exist: " + data_folder << std::endl;
            return 1;
        }

        int set_id = db.createKeypointSetWithOverlap(
            set_name,
            detector_str,
            "non_overlapping_detection",
            2000,
            data_folder,
            detector_str + " detector with non-overlapping constraint (min_distance=" + std::to_string(min_distance) + "px)",
            40,
            true,
            min_distance
        );

        if (set_id == -1) {
            std::cerr << "Failed to create keypoint set: " + set_name << std::endl;
            return 1;
        }

        LOG_INFO("Created keypoint set with ID: " + std::to_string(set_id));

        auto detector = KeypointGeneratorFactory::create(detector_type, true, min_distance);
        KeypointParams params;
        params.max_features = 2000;

        int total_keypoints = 0;
        int images_processed = 0;
        double detection_time_ms = 0.0;
        const auto generation_start = std::chrono::steady_clock::now();

        for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
            if (!fs::is_directory(scene_entry)) continue;

            std::string scene_name = scene_entry.path().filename().string();
            LOG_INFO("Processing scene: " + scene_name);

            for (int i = 1; i <= 6; ++i) {
                fs::path image_path = scene_entry.path() / (std::to_string(i) + ".ppm");
                if (!fs::exists(image_path)) {
                    std::cerr << "Image not found: " << image_path.string() << std::endl;
                    continue;
                }

                cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) {
                    std::cerr << "Could not load image: " << image_path.string() << std::endl;
                    continue;
                }

                const auto detect_start = std::chrono::steady_clock::now();
                std::vector<cv::KeyPoint> keypoints = detector->detectNonOverlapping(image, min_distance, params);
                const auto detect_end = std::chrono::steady_clock::now();
                detection_time_ms += std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
                ++images_processed;

                std::string image_name = std::to_string(i) + ".ppm";
                if (db.storeLockedKeypointsForSet(set_id, scene_name, image_name, keypoints)) {
                    total_keypoints += static_cast<int>(keypoints.size());
                    LOG_INFO(scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                } else {
                    LOG_ERROR("Failed to store keypoints for " + scene_name + "/" + image_name);
                }
            }
        }

        LOG_INFO("Generation complete. Non-overlapping " + detector_str + " keypoints stored in set: " + set_name);
        LOG_INFO("Total keypoints generated: " + std::to_string(total_keypoints));

        const auto generation_end = std::chrono::steady_clock::now();
        const double generation_time_ms = std::chrono::duration<double, std::milli>(generation_end - generation_start).count();
        thesis_project::database::KeypointSetStats stats;
        stats.keypoint_set_id = set_id;
        stats.detection_time_cpu_ms = detection_time_ms;
        stats.total_generation_cpu_ms = generation_time_ms;
        stats.total_keypoints = total_keypoints;
        stats.avg_keypoints_per_image = images_processed > 0
            ? static_cast<double>(total_keypoints) / static_cast<double>(images_processed)
            : 0.0;
        if (!db.updateKeypointSetStats(stats)) {
            LOG_WARNING("Failed to persist keypoint generation stats for set " + set_name);
        } else {
            std::ostringstream stats_stream;
            stats_stream << std::fixed << std::setprecision(2)
                         << "Generation stats -> detection_cpu_ms=" << detection_time_ms
                         << ", avg_kp_per_image=" << stats.avg_keypoints_per_image;
            LOG_INFO(stats_stream.str());
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Error: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
