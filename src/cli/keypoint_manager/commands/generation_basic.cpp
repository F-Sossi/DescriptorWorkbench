#include "generation_basic.hpp"
#include "src/core/keypoints/locked_in_keypoints.hpp"
#include "thesis_project/logging.hpp"
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iostream>

namespace thesis_project::cli::keypoint_commands {

int generateProjected(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " generate-projected <data_folder> [name]" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-projected ../data" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-projected ../data my_projected_keypoints" << std::endl;
        return 1;
    }

    std::string data_folder = argv[2];
    std::string set_name = (argc == 4) ? argv[3] : "homography_projection_" + std::to_string(std::time(nullptr));

    LOG_INFO("Generating homography projected keypoints from: " + data_folder);
    LOG_INFO("Keypoint set name: " + set_name);

    namespace fs = boost::filesystem;
    if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
        std::cerr << "Data folder does not exist: " + data_folder << std::endl;
        return 1;
    }

    int set_id = db.createKeypointSet(
        set_name,
        "SIFT",
        "homography_projection",
        2000,
        data_folder,
        "Homography projected keypoints with 40px boundary filtering",
        40
    );

    if (set_id == -1) {
        std::cerr << "Failed to create keypoint set: " + set_name << std::endl;
        return 1;
    }

    LOG_INFO("Created keypoint set with ID: " + std::to_string(set_id));
    LOG_INFO("🔍 Generating keypoints with homography projection and boundary filtering...");

    try {
        LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db, set_id, 2000);
        LOG_INFO("🎉 Generation complete! Homography projected keypoints stored in set: " + set_name);
    } catch (const std::exception& e) {
            LOG_ERROR("Error generating keypoints: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

int generateIndependent(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " generate-independent <data_folder> [name]" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-independent ../data" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-independent ../data my_independent_keypoints" << std::endl;
        return 1;
    }

    std::string data_folder = argv[2];
    std::string set_name = (argc == 4) ? argv[3] : "independent_detection_" + std::to_string(std::time(nullptr));

    LOG_INFO("Generating independent detection keypoints from: " + data_folder);
    LOG_INFO("Keypoint set name: " + set_name);

    namespace fs = boost::filesystem;
    if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
        std::cerr << "Data folder does not exist: " + data_folder << std::endl;
        return 1;
    }

    int set_id = db.createKeypointSet(
        set_name,
        "SIFT",
        "independent_detection",
        2000,
        data_folder,
        "Independent SIFT detection on each image with 40px boundary filtering",
        40
    );

    if (set_id == -1) {
        std::cerr << "Failed to create keypoint set: " + set_name << std::endl;
        return 1;
    }

    LOG_INFO("Created keypoint set with ID: " + std::to_string(set_id));
    LOG_INFO("🔍 Generating keypoints with independent detection on each image...");

    try {
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        const int BORDER = 40;
        int total_keypoints = 0;
        int images_processed = 0;
        double detection_time_ms = 0.0;
        const auto generation_start = std::chrono::steady_clock::now();

        for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
            if (!fs::is_directory(scene_entry)) continue;

            std::string scene_name = scene_entry.path().filename().string();
            LOG_INFO("📁 Processing scene: " + scene_name);

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

                std::vector<cv::KeyPoint> keypoints;
                const auto detect_start = std::chrono::steady_clock::now();
                detector->detect(image, keypoints);
                const auto detect_end = std::chrono::steady_clock::now();
                detection_time_ms += std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

                keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image, BORDER](const cv::KeyPoint& keypoint) {
                    return keypoint.pt.x < BORDER || keypoint.pt.y < BORDER ||
                           keypoint.pt.x > (image.cols - BORDER) || keypoint.pt.y > (image.rows - BORDER);
                }), keypoints.end());

                if (keypoints.size() > 2000) {
                    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                        return a.response > b.response;
                    });
                    keypoints.resize(2000);
                }

                std::string image_name = std::to_string(i) + ".ppm";
                if (db.storeLockedKeypointsForSet(set_id, scene_name, image_name, keypoints)) {
                    total_keypoints += static_cast<int>(keypoints.size());
                    ++images_processed;
                        LOG_INFO("Processed " + scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                } else {
                        LOG_ERROR("Failed to store keypoints for " + scene_name + "/" + image_name);
                }
            }
        }

        LOG_INFO("Generation complete! Independent detection keypoints stored in set: " + set_name);
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
                             << "Generation stats: detection_cpu_ms=" << detection_time_ms
                             << ", avg_kp_per_image=" << stats.avg_keypoints_per_image;
                LOG_INFO(stats_stream.str());
            }

    } catch (const std::exception& e) {
            LOG_ERROR("Error generating keypoints: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

int generateLegacy(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " generate <data_folder>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate ../data" << std::endl;
        return 1;
    }

    std::string data_folder = argv[2];
    LOG_INFO("Generating fresh locked keypoints from: " + data_folder);

    namespace fs = boost::filesystem;
    if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
        std::cerr << "Data folder does not exist: " + data_folder << std::endl;
        return 1;
    }

    LOG_INFO("Clearing existing keypoints from database...");
    for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
        if (!fs::is_directory(scene_entry)) continue;
        std::string scene_name = scene_entry.path().filename().string();
        if (!db.clearSceneKeypoints(scene_name)) {
                LOG_ERROR("Failed to clear keypoints for scene: " + scene_name);
            return 1;
        }
    }

    LOG_INFO("Generating new locked keypoints with proper boundary filtering...");

    try {
        LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db);
        LOG_INFO("Generation complete! Keypoints generated with 40px boundary filtering.");
    } catch (const std::exception& e) {
            LOG_ERROR("Error generating keypoints: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
