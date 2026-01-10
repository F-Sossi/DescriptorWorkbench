#include "import_export.hpp"
#include "src/core/keypoints/locked_in_keypoints.hpp"
#include "thesis_project/logging.hpp"
#include <boost/filesystem.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace thesis_project::cli::keypoint_commands {

int importCsv(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " import-csv <csv_folder>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " import-csv ../reference_keypoints" << std::endl;
        return 1;
    }

    std::string csv_folder = argv[2];
    LOG_INFO("Importing keypoints from CSV folder: " + csv_folder);

    namespace fs = boost::filesystem;

    if (!fs::exists(csv_folder) || !fs::is_directory(csv_folder)) {
        std::cerr << "CSV folder does not exist: " << csv_folder << std::endl;
        return 1;
    }

    int total_imported = 0;

    // Iterate through each scene folder
    for (const auto& scene_entry : fs::directory_iterator(csv_folder)) {
        if (!fs::is_directory(scene_entry)) continue;

        std::string scene_name = scene_entry.path().filename().string();
        LOG_INFO("Processing scene: " + scene_name);

        // Iterate through each CSV file in the scene folder
        for (const auto& csv_entry : fs::directory_iterator(scene_entry)) {
            if (csv_entry.path().extension() != ".csv") continue;

            std::string csv_file = csv_entry.path().string();
            std::string image_name = csv_entry.path().stem().string() + ".ppm";

            try {
                std::vector<cv::KeyPoint> keypoints = LockedInKeypoints::readKeypointsFromCSV(csv_file);

                    if (db.storeLockedKeypoints(scene_name, image_name, keypoints)) {
                        total_imported += static_cast<int>(keypoints.size());
                        LOG_INFO(scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("Failed to store: " + scene_name + "/" + image_name);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Error processing " + csv_file + ": " + std::string(e.what()));
                }
            }
        }

    LOG_INFO("Import complete! Total keypoints imported: " + std::to_string(total_imported));
    return 0;
}

int exportCsv(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " export-csv <output_folder>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " export-csv ./exported_keypoints" << std::endl;
        return 1;
    }

    std::string output_folder = argv[2];
    LOG_INFO("Exporting keypoints to CSV folder: " + output_folder);

    // Create output directory
    std::filesystem::create_directories(output_folder);

    auto scenes = db.getAvailableScenes();
    if (scenes.empty()) {
        LOG_INFO("No keypoints found in database to export");
        return 0;
    }

    int total_exported = 0;

    for (const auto& scene : scenes) {
        std::string scene_folder = output_folder + "/" + scene;
        std::filesystem::create_directories(scene_folder);

        LOG_INFO("📁 Exporting scene: " + scene);

        auto images = db.getAvailableImages(scene);
        for (const auto& image : images) {
            auto keypoints = db.getLockedKeypoints(scene, image);

            if (!keypoints.empty()) {
                // Convert .ppm to .csv filename
                std::string csv_filename = image;
                if (csv_filename.length() >= 4 && csv_filename.substr(csv_filename.length() - 4) == ".ppm") {
                    csv_filename = csv_filename.substr(0, csv_filename.length() - 4) + "ppm.csv";
                } else {
                    csv_filename += ".csv";
                }

                std::string csv_path = scene_folder + "/" + csv_filename;

                // Write CSV file
                std::ofstream file(csv_path);
                if (!file.is_open()) {
                    LOG_ERROR("Failed to create file: " + csv_path);
                    continue;
                }

                // Write header
                file << "x,y,size,angle,response,octave,class_id" << std::endl;

                // Write keypoints
                for (const auto& kp : keypoints) {
                    file << kp.pt.x << "," << kp.pt.y << "," << kp.size << ","
                         << kp.angle << "," << kp.response << "," << kp.octave << ","
                         << kp.class_id << std::endl;
                }

                file.close();
                total_exported += static_cast<int>(keypoints.size());
                LOG_INFO(scene + "/" + csv_filename + ": " + std::to_string(keypoints.size()) + " keypoints");
            }
        }
    }

    LOG_INFO("Export complete Total keypoints exported: " + std::to_string(total_exported));
    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
