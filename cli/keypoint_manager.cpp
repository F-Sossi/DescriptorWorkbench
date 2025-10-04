#include "thesis_project/database/DatabaseManager.hpp"
#include "src/core/keypoints/locked_in_keypoints.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include "src/core/processing/processor_utils.hpp"
#include "thesis_project/logging.hpp"
#include "thesis_project/types.hpp"
#include <iostream>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>

namespace cv {
    class KeyPoint;
}

using namespace thesis_project;

/**
 * @brief CLI tool for managing locked-in keypoints in the database
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <command> [options]" << std::endl;
        std::cout << "Commands:" << std::endl;
        std::cout << "  Keypoint Generation:" << std::endl;
        std::cout << "    generate-projected <data_folder> [name]   - Generate keypoints using homography projection (controlled)" << std::endl;
        std::cout << "    generate-independent <data_folder> [name] - Generate keypoints using independent detection (realistic)" << std::endl;
        std::cout << "    generate <data_folder>                    - Legacy: Generate homography projected keypoints" << std::endl;
        std::cout << "  Advanced Detector Generation:" << std::endl;
        std::cout << "    generate-detector <data_folder> <detector> [name]                    - Generate keypoints using specific detector (sift|harris|orb)" << std::endl;
        std::cout << "    generate-non-overlapping <data_folder> <detector> <min_distance> [name] - Generate non-overlapping keypoints" << std::endl;
        std::cout << "    generate-kornia-keynet <data_folder> [set_name] [max_kp] [device] [--mode independent|projected] [--overwrite]" << std::endl;
        std::cout << "                         Run Kornia KeyNet detector via Python (independent or homography projected)" << std::endl;
        // attribute-detector removed - using pure intersection sets instead
        std::cout << "    build-intersection --source-a <set> --source-b <set> --out-a <set> --out-b <set> [--tolerance px] [--overwrite]" << std::endl;
        std::cout << "                         Create paired subsets where two detectors agree spatially" << std::endl;
        std::cout << "  Import/Export:" << std::endl;
        std::cout << "    import-csv <csv_folder> [set_name]        - Import keypoints from CSV files" << std::endl;
        std::cout << "    export-csv <output_folder> [set_id]       - Export keypoints from DB to CSV" << std::endl;
        std::cout << "  Information:" << std::endl;
        std::cout << "    list-sets                                 - List all available keypoint sets" << std::endl;
        std::cout << "    list-scenes [set_id]                      - List scenes in database (optionally for specific set)" << std::endl;
        std::cout << "    count <scene> <image> [set_id]            - Count keypoints for specific scene/image" << std::endl;
        std::cout << "    list-detectors                            - List supported detector types" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    // Initialize database
    database::DatabaseManager db("experiments.db", true);
    if (!db.isEnabled()) {
        std::cerr << "❌ Failed to connect to database" << std::endl;
        return 1;
    }
    
    // Optimize database for bulk operations
    if (!db.optimizeForBulkOperations()) {
        std::cerr << "Warning: Failed to apply database optimizations" << std::endl;
    }

    if (command == "import-csv") {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " import-csv <csv_folder>" << std::endl;
            std::cerr << "  Example: " << argv[0] << " import-csv ../reference_keypoints" << std::endl;
            return 1;
        }

        std::string csv_folder = argv[2];
        LOG_INFO("Importing keypoints from CSV folder: " + csv_folder);

        namespace fs = boost::filesystem;
        
        if (!fs::exists(csv_folder) || !fs::is_directory(csv_folder)) {
            std::cerr << "❌ CSV folder does not exist: " << csv_folder << std::endl;
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
                        total_imported += keypoints.size();
                        LOG_INFO( scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("  ❌ Failed to store: " + scene_name + "/" + image_name);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("  ❌ Error processing " + csv_file + ": " + std::string(e.what()));
                }
            }
        }
        
        LOG_INFO("Import complete! Total keypoints imported: " + std::to_string(total_imported));

    } else if (command == "generate-projected") {
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
            std::cerr << "❌ Data folder does not exist: " + data_folder << std::endl;
            return 1;
        }

        // Create keypoint set
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
            std::cerr << "❌ Failed to create keypoint set: " + set_name << std::endl;
            return 1;
        }
        
        LOG_INFO("Created keypoint set with ID: " + std::to_string(set_id));
        
        // Generate keypoints using homography projection method
        LOG_INFO("🔍 Generating keypoints with homography projection and boundary filtering...");
        
        try {
            // Use the existing LockedInKeypoints method but store to specific set
            LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db, set_id, 2000);
            LOG_INFO("🎉 Generation complete! Homography projected keypoints stored in set: " + set_name);
            
        } catch (const std::exception& e) {
            LOG_ERROR("❌ Error generating keypoints: " + std::string(e.what()));
            return 1;
        }

    } else if (command == "generate-independent") {
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
            std::cerr << "❌ Data folder does not exist: " + data_folder << std::endl;
            return 1;
        }

        // Create keypoint set
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
            std::cerr << "❌ Failed to create keypoint set: " + set_name << std::endl;
            return 1;
        }
        
        LOG_INFO("Created keypoint set with ID: " + std::to_string(set_id));
        
        // Generate keypoints using independent detection method
        LOG_INFO("🔍 Generating keypoints with independent detection on each image...");
        
        try {
            cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
            const int BORDER = 40;
            int total_keypoints = 0;

            for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
                if (!fs::is_directory(scene_entry)) continue;
                
                std::string scene_name = scene_entry.path().filename().string();
                LOG_INFO("📁 Processing scene: " + scene_name);
                
                // Process each image independently (1.ppm to 6.ppm)
                for (int i = 1; i <= 6; ++i) {
                    fs::path image_path = scene_entry.path() / (std::to_string(i) + ".ppm");
                    if (!fs::exists(image_path)) {
                        std::cerr << "❌ Image not found: " << image_path.string() << std::endl;
                        continue;
                    }
                    
                    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
                    if (image.empty()) {
                        std::cerr << "❌ Could not load image: " << image_path.string() << std::endl;
                        continue;
                    }
                    
                    // Detect keypoints independently on this image
                    std::vector<cv::KeyPoint> keypoints;
                    detector->detect(image, keypoints);
                    
                    // Apply boundary filtering
                    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image, BORDER](const cv::KeyPoint& keypoint) {
                        return keypoint.pt.x < BORDER || keypoint.pt.y < BORDER ||
                               keypoint.pt.x > (image.cols - BORDER) || keypoint.pt.y > (image.rows - BORDER);
                    }), keypoints.end());
                    
                    // Limit to 2000 keypoints (sorted by response strength)
                    if (keypoints.size() > 2000) {
                        std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                            return a.response > b.response;
                        });
                        keypoints.resize(2000);
                    }
                    
                    // Store keypoints for this image
                    std::string image_name = std::to_string(i) + ".ppm";
                    if (db.storeLockedKeypointsForSet(set_id, scene_name, image_name, keypoints)) {
                        total_keypoints += keypoints.size();
                        LOG_INFO("  ✅ " + scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("  ❌ Failed to store keypoints for " + scene_name + "/" + image_name);
                    }
                }
            }
            
            LOG_INFO("Generation complete! Independent detection keypoints stored in set: " + set_name);
            LOG_INFO("Total keypoints generated: " + std::to_string(total_keypoints));
            
        } catch (const std::exception& e) {
            LOG_ERROR("❌ Error generating keypoints: " + std::string(e.what()));
            return 1;
        }

    } else if (command == "generate") {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " generate <data_folder>" << std::endl;
            std::cerr << "  Example: " << argv[0] << " generate ../data" << std::endl;
            return 1;
        }
        
        std::string data_folder = argv[2];
        LOG_INFO("Generating fresh locked keypoints from: " + data_folder);
        
        namespace fs = boost::filesystem;
        if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
            std::cerr << "❌ Data folder does not exist: " << data_folder << std::endl;
            return 1;
        }

        // Clear existing keypoints for all scenes in the data folder
        LOG_INFO("Clearing existing keypoints from database...");
        for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
            if (!fs::is_directory(scene_entry)) continue;
            std::string scene_name = scene_entry.path().filename().string();
            if (!db.clearSceneKeypoints(scene_name)) {
                LOG_ERROR("❌ Failed to clear keypoints for scene: " + scene_name);
                return 1;
            }
        }
        
        // Generate fresh keypoints using tested boundary-filtering logic
        LOG_INFO("Generating new locked keypoints with proper boundary filtering...");
        
        try {
            LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db);
            LOG_INFO("Generation complete! Keypoints generated with 40px boundary filtering.");
            
        } catch (const std::exception& e) {
            LOG_ERROR("❌ Error generating keypoints: " + std::string(e.what()));
            return 1;
        }

    } else if (command == "export-csv") {
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
                        LOG_ERROR("  ❌ Failed to create file: " + csv_path);
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
                    total_exported += keypoints.size();
                    LOG_INFO( scene + "/" + csv_filename + ": " + std::to_string(keypoints.size()) + " keypoints");
                }
            }
        }
        
        LOG_INFO("Export complete Total keypoints exported: " + std::to_string(total_exported));

    } else if (command == "list-sets") {
        auto sets = db.getAvailableKeypointSets();
        std::cout << "Available keypoint sets (" << sets.size() << "):" << std::endl;
        for (const auto& [id, name, method] : sets) {
            // Count keypoints in this set
            std::string sql = "SELECT COUNT(*) FROM locked_keypoints WHERE keypoint_set_id = " + std::to_string(id);
            // For now, let's show the basic info - we can add counts later if needed
            std::cout << "ID " << id << ": " << name << " (" << method << ")" << std::endl;
        }
        if (sets.empty()) {
            std::cout << "  (No keypoint sets found - use generate-projected or generate-independent to create some)" << std::endl;
        }

    } else if (command == "list-scenes") {
        auto scenes = db.getAvailableScenes();
        std::cout << "📋 Available scenes (" << scenes.size() << "):" << std::endl;
        for (const auto& scene : scenes) {
            auto images = db.getAvailableImages(scene);
            int total_keypoints = 0;
            for (const auto& image : images) {
                auto keypoints = db.getLockedKeypoints(scene, image);
                total_keypoints += keypoints.size();
            }
            std::cout << "  📁 " << scene << " (" << images.size() << " images, " << total_keypoints << " total keypoints)" << std::endl;
        }

    } else if (command == "count") {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " count <scene> <image>" << std::endl;
            std::cerr << "  Example: " << argv[0] << " count i_dome 1.ppm" << std::endl;
            return 1;
        }
        
        std::string scene = argv[2];
        std::string image = argv[3];
        auto keypoints = db.getLockedKeypoints(scene, image);
        std::cout << "🔢 Keypoints for " << scene << "/" << image << ": " << keypoints.size() << std::endl;

    } else if (command == "generate-kornia-keynet") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " generate-kornia-keynet <data_folder> [set_name] [max_kp] [device] [--overwrite]" << std::endl;
            std::cerr << "  Example: " << argv[0] << " generate-kornia-keynet ../data keynet_reference 2000 auto --overwrite" << std::endl;
            return 1;
        }

        std::string data_folder = argv[2];
        std::string set_name = (argc >= 4 && std::string(argv[3]).rfind("--", 0) != 0)
            ? argv[3]
            : ("keynet_kornia_" + std::to_string(std::time(nullptr)));

        int arg_index = 4;
        int max_kp = 2000;
        if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
            max_kp = std::stoi(argv[arg_index]);
            ++arg_index;
        }

        std::string device_arg = "auto";
        if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
            device_arg = argv[arg_index];
            ++arg_index;
        }

        std::string mode_arg = "independent";
        bool overwrite = false;
        while (arg_index < argc) {
            std::string extra = argv[arg_index];
            if (extra == "--overwrite") {
                overwrite = true;
                ++arg_index;
            } else if (extra == "--mode" && (arg_index + 1) < argc) {
                mode_arg = argv[arg_index + 1];
                arg_index += 2;
            } else {
                std::cerr << "Unknown option for generate-kornia-keynet: " << extra << std::endl;
                return 1;
            }
        }

        if (mode_arg != "independent" && mode_arg != "projected") {
            std::cerr << "Unsupported mode for Kornia KeyNet: " << mode_arg << " (expected 'independent' or 'projected')" << std::endl;
            return 1;
        }

        LOG_INFO("Launching Kornia KeyNet generation via Python");
        LOG_INFO("  Data folder: " + data_folder);
        LOG_INFO("  Set name: " + set_name);
        LOG_INFO("  Max keypoints: " + std::to_string(max_kp));
        LOG_INFO("  Device: " + device_arg);
        LOG_INFO(std::string("  Mode: ") + mode_arg);
        LOG_INFO(std::string("  Overwrite: ") + (overwrite ? "true" : "false"));

        std::ostringstream cmd;
        cmd << "/bin/bash -c \"source /home/frank/miniforge3/etc/profile.d/conda.sh && conda activate descriptor-compare && "
            << "python3 ../scripts/generate_keynet_keypoints.py"
            << " --data_dir \"" << data_folder << "\""
            << " --db_path experiments.db"
            << " --max_keypoints " << max_kp
            << " --device " << device_arg
            << " --mode " << mode_arg
            << " --set-name \"" << set_name << "\"";
        if (overwrite) {
            cmd << " --overwrite";
        }
        cmd << "\"";

        int result = std::system(cmd.str().c_str());
        if (result != 0) {
            LOG_ERROR("Kornia KeyNet generation failed. Exit code: " + std::to_string(result));
            return 1;
        }

        LOG_INFO("Kornia KeyNet generation completed successfully for set: " + set_name);

    } else if (command == "generate-detector") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " generate-detector <data_folder> <detector> [name] [--max-features N] [--overwrite]" << std::endl;
            std::cerr << "  Detectors: sift, harris, orb" << std::endl;
            return 1;
        }

        std::string data_folder = argv[2];
        std::string detector_str = argv[3];
        int arg_index = 4;
        std::string set_name;
        if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
            set_name = argv[arg_index++];
        } else {
            set_name = detector_str + "_keypoints_" + std::to_string(std::time(nullptr));
        }

        int max_features = 2000;
        bool overwrite = false;

        while (arg_index < argc) {
            std::string arg = argv[arg_index++];
            if (arg == "--max-features") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --max-features" << std::endl;
                    return 1;
                }
                max_features = std::stoi(argv[arg_index++]);
            } else if (arg == "--overwrite") {
                overwrite = true;
            } else {
                std::cerr << "❌ Unknown option: " << arg << std::endl;
                return 1;
            }
        }

        try {
            KeypointGenerator detector_type = KeypointGeneratorFactory::parseDetectorType(detector_str);

            namespace fs = boost::filesystem;
            if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
                std::cerr << "❌ Data folder does not exist: " << data_folder << std::endl;
                return 1;
            }

            LOG_INFO("Generating keypoints using " + detector_str + " detector");
            LOG_INFO("Data folder: " + data_folder);
            LOG_INFO("Keypoint set name: " + set_name);
            LOG_INFO("Max features per image: " + std::to_string(max_features > 0 ? max_features : -1));

            int set_id = db.getKeypointSetId(set_name);
            if (set_id >= 0) {
                if (!overwrite) {
                    std::cerr << "❌ Keypoint set already exists: " << set_name << " (use --overwrite to replace)" << std::endl;
                    return 1;
                }

                if (db.clearAllDetectorAttributesForSet(set_id)) {
                    LOG_INFO("database clear all detector attributes for set: " + set_name);
                }else {
                    LOG_INFO("Database clear all detector attributes for set: " + set_name + " failed");
                }

                if (db.clearKeypointsForSet(set_id)) {
                    LOG_INFO("Keypoint clear all keypoints for set: " + set_name);
                }else {
                    LOG_INFO("Keypoint clear failed for set: " + set_name);
                }

            } else {
                std::ostringstream desc;
                desc << detector_str << " detector (independent)";
                set_id = db.createKeypointSet(
                    set_name,
                    detector_str,
                    "independent_detection",
                    max_features,
                    data_folder,
                    desc.str(),
                    40
                );

                if (set_id == -1) {
                    std::cerr << "❌ Failed to create keypoint set: " << set_name << std::endl;
                    return 1;
                }
            }

            auto detector = KeypointGeneratorFactory::create(detector_type, false, 0.0f);
            KeypointParams params;
            params.max_features = max_features;

            auto trimByResponse = [](std::vector<cv::KeyPoint>& keypoints, int limit) {
                if (limit > 0 && keypoints.size() > static_cast<size_t>(limit)) {
                    std::nth_element(keypoints.begin(), keypoints.begin() + limit, keypoints.end(),
                        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                            return a.response > b.response;
                        });
                    keypoints.resize(limit);
                    std::sort(keypoints.begin(), keypoints.end(),
                        [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                            return a.response > b.response;
                        });
                }
            };

            int total_keypoints = 0;

            for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
                if (!fs::is_directory(scene_entry)) continue;
                std::string scene_name = scene_entry.path().filename().string();
                LOG_INFO("Processing scene: " + scene_name);

                for (int i = 1; i <= 6; ++i) {
                    fs::path image_path = scene_entry.path() / (std::to_string(i) + ".ppm");
                    if (!fs::exists(image_path)) {
                        std::cerr << "❌ Image not found: " << image_path.string() << std::endl;
                        continue;
                    }

                    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
                    if (image.empty()) {
                        std::cerr << "❌ Could not load image: " << image_path.string() << std::endl;
                        continue;
                    }

                    std::vector<cv::KeyPoint> keypoints = detector->detect(image, params);
                    trimByResponse(keypoints, max_features);

                    std::string image_name = std::to_string(i) + ".ppm";
                    if (db.storeLockedKeypointsForSet(set_id, scene_name, image_name, keypoints)) {
                        total_keypoints += static_cast<int>(keypoints.size());
                        LOG_INFO(scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("  ❌ Failed to store keypoints for " + scene_name + "/" + image_name);
                    }
                }
            }

            LOG_INFO("Generation complete! " + detector_str + " keypoints stored in set: " + set_name);
            LOG_INFO("Total keypoints generated: " + std::to_string(total_keypoints));

        } catch (const std::exception& e) {
            LOG_ERROR("❌ Error: " + std::string(e.what()));
            return 1;
        }

    } else if (command == "build-intersection") {
        std::string source_a_name;
        std::string source_b_name;
        std::string output_a_name;
        std::string output_b_name;
        double tolerance_px = 3.0;
        bool overwrite = false;

        auto usage = [&]() {
            std::cerr << "Usage: " << argv[0]
                      << " build-intersection --source-a <set> --source-b <set> --out-a <set> --out-b <set> [--tolerance px] [--overwrite]"
                      << std::endl;
        };

        int arg_index = 2;
        while (arg_index < argc) {
            std::string arg = argv[arg_index++];
            if (arg == "--source-a") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --source-a" << std::endl;
                    return 1;
                }
                source_a_name = argv[arg_index++];
            } else if (arg == "--source-b") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --source-b" << std::endl;
                    return 1;
                }
                source_b_name = argv[arg_index++];
            } else if (arg == "--out-a") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --out-a" << std::endl;
                    return 1;
                }
                output_a_name = argv[arg_index++];
            } else if (arg == "--out-b") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --out-b" << std::endl;
                    return 1;
                }
                output_b_name = argv[arg_index++];
            } else if (arg == "--tolerance") {
                if (arg_index >= argc) {
                    std::cerr << "❌ Missing value for --tolerance" << std::endl;
                    return 1;
                }
                tolerance_px = std::stod(argv[arg_index++]);
            } else if (arg == "--overwrite") {
                overwrite = true;
            } else {
                std::cerr << "❌ Unknown option: " << arg << std::endl;
                usage();
                return 1;
            }
        }

        if (source_a_name.empty() || source_b_name.empty() || output_a_name.empty() || output_b_name.empty()) {
            usage();
            return 1;
        }

        if (tolerance_px <= 0.0) {
            std::cerr << "❌ Tolerance must be positive" << std::endl;
            return 1;
        }

        auto source_a_info = db.getKeypointSetInfo(source_a_name);
        if (!source_a_info) {
            std::cerr << "❌ Source keypoint set not found: " << source_a_name << std::endl;
            return 1;
        }

        auto source_b_info = db.getKeypointSetInfo(source_b_name);
        if (!source_b_info) {
            std::cerr << "❌ Source keypoint set not found: " << source_b_name << std::endl;
            return 1;
        }

        if (!source_a_info->dataset_path.empty() && !source_b_info->dataset_path.empty() &&
            source_a_info->dataset_path != source_b_info->dataset_path) {
            LOG_WARNING("Source sets reference different datasets: " + source_a_info->dataset_path + " vs " + source_b_info->dataset_path);
        }

        // Pure intersection approach - no detector attribute copying needed

        auto prepareOutputSet = [&](const database::DatabaseManager::KeypointSetInfo& source_info,
                                    const std::string& output_name,
                                    const std::string& partner_name,
                                    int partner_set_id) -> std::optional<int> {
            int existing_id = db.getKeypointSetId(output_name);

            std::ostringstream desc;
            desc << "Paired subset from " << source_info.name << " matched with " << partner_name
                 << " (" << tolerance_px << "px tolerance)";

            if (existing_id >= 0) {
                if (!overwrite) {
                    std::cerr << "❌ Output set already exists: " << output_name << " (use --overwrite to replace)" << std::endl;
                    return std::nullopt;
                }
                if (!db.clearAllDetectorAttributesForSet(existing_id)) {
                    LOG_WARNING("⚠️  Failed to clear detector attributes for " + output_name + ", proceeding with keypoint overwrite");
                }
                if (!db.clearKeypointsForSet(existing_id)) {
                    std::cerr << "❌ Failed to clear existing keypoints for set: " << output_name << std::endl;
                    return std::nullopt;
                }
                if (!db.updateIntersectionKeypointSet(existing_id,
                                                       source_info.generator_type,
                                                       source_info.generation_method,
                                                       source_info.max_features,
                                                       source_info.dataset_path,
                                                       desc.str(),
                                                       source_info.boundary_filter_px,
                                                       source_info.id,
                                                       partner_set_id,
                                                       static_cast<float>(tolerance_px),
                                                       "mutual_nearest_neighbor")) {
                    std::cerr << "❌ Failed to update metadata for existing intersection set: " << output_name << std::endl;
                    return std::nullopt;
                }
                return existing_id;
            }

            int new_id = db.createIntersectionKeypointSet(
                output_name,
                source_info.generator_type,
                source_info.generation_method,
                source_info.max_features,
                source_info.dataset_path,
                desc.str(),
                source_info.boundary_filter_px,
                source_info.id,           // source_set_a_id
                partner_set_id,           // source_set_b_id
                static_cast<float>(tolerance_px),             // tolerance_px
                "mutual_nearest_neighbor" // intersection_method
            );

            if (new_id == -1) {
                std::cerr << "❌ Failed to create output keypoint set: " << output_name << std::endl;
                return std::nullopt;
            }

            return new_id;
        };

        auto output_a_id_opt = prepareOutputSet(*source_a_info, output_a_name, source_b_info->name, source_b_info->id);
        if (!output_a_id_opt) {
            return 1;
        }

        auto output_b_id_opt = prepareOutputSet(*source_b_info, output_b_name, source_a_info->name, source_a_info->id);
        if (!output_b_id_opt) {
            return 1;
        }

        int output_a_id = *output_a_id_opt;
        int output_b_id = *output_b_id_opt;

        auto scenes_a = db.getScenesForSet(source_a_info->id);
        auto scenes_b = db.getScenesForSet(source_b_info->id);

        std::unordered_set<std::string> scene_lookup_b(scenes_b.begin(), scenes_b.end());
        std::vector<std::string> scenes_to_process;
        scenes_to_process.reserve(scenes_a.size());

        for (const auto& scene : scenes_a) {
            if (scene_lookup_b.count(scene)) {
                scenes_to_process.push_back(scene);
            } else {
                LOG_WARNING("Skipping scene " + scene + " (missing in " + source_b_info->name + ")");
            }
        }

        if (scenes_to_process.empty()) {
            std::cerr << "❌ No overlapping scenes between " << source_a_name << " and " << source_b_name << std::endl;
            return 1;
        }

        const float tolerance_sq = static_cast<float>(tolerance_px * tolerance_px);
        cv::flann::SearchParams search_params(32);

        size_t total_pairs = 0;
        size_t total_inserted_a = 0;
        size_t total_inserted_b = 0;
        size_t total_candidates_a = 0;
        size_t total_candidates_b = 0;
        size_t processed_images = 0;
        size_t skipped_images = 0;

        for (const auto& scene : scenes_to_process) {
            auto images_a = db.getImagesForSet(source_a_info->id, scene);
            auto images_b = db.getImagesForSet(source_b_info->id, scene);

            std::unordered_set<std::string> image_lookup_b(images_b.begin(), images_b.end());

            for (const auto& image : images_a) {
                if (!image_lookup_b.count(image)) {
                    ++skipped_images;
                    LOG_WARNING("⚠️  Skipping " + scene + "/" + image + " (missing in " + source_b_info->name + ")");
                    continue;
                }

                auto records_a = db.getLockedKeypointsWithIds(source_a_info->id, scene, image);
                auto records_b = db.getLockedKeypointsWithIds(source_b_info->id, scene, image);

                if (records_a.empty() || records_b.empty()) {
                    ++skipped_images;
                    LOG_WARNING("⚠️  " + scene + "/" + image + ": no keypoints to match (" +
                             std::to_string(records_a.size()) + " from A, " + std::to_string(records_b.size()) + " from B)");
                    continue;
                }

                ++processed_images;
                total_candidates_a += records_a.size();
                total_candidates_b += records_b.size();

                cv::Mat points_a(static_cast<int>(records_a.size()), 2, CV_32F);
                cv::Mat points_b(static_cast<int>(records_b.size()), 2, CV_32F);

                for (size_t i = 0; i < records_a.size(); ++i) {
                    points_a.at<float>(static_cast<int>(i), 0) = static_cast<float>(records_a[i].keypoint.pt.x);
                    points_a.at<float>(static_cast<int>(i), 1) = static_cast<float>(records_a[i].keypoint.pt.y);
                }

                for (size_t i = 0; i < records_b.size(); ++i) {
                    points_b.at<float>(static_cast<int>(i), 0) = static_cast<float>(records_b[i].keypoint.pt.x);
                    points_b.at<float>(static_cast<int>(i), 1) = static_cast<float>(records_b[i].keypoint.pt.y);
                }

                cv::flann::Index index_a(points_a, cv::flann::KDTreeIndexParams(4));
                cv::flann::Index index_b(points_b, cv::flann::KDTreeIndexParams(4));

                std::vector<std::pair<size_t, size_t>> matches;
                matches.reserve(std::min(records_a.size(), records_b.size()));

                std::vector<int> nn_index(1);
                std::vector<float> nn_dist(1);
                std::vector<int> reverse_index(1);
                std::vector<float> reverse_dist(1);
                std::vector<char> used_b(records_b.size(), 0);

                for (size_t a_idx = 0; a_idx < records_a.size(); ++a_idx) {
                    cv::Mat query_a = points_a.row(static_cast<int>(a_idx));
                    index_b.knnSearch(query_a, nn_index, nn_dist, 1, search_params);

                    int b_idx = nn_index[0];
                    if (b_idx < 0 || static_cast<size_t>(b_idx) >= records_b.size()) {
                        continue;
                    }

                    if (nn_dist[0] > tolerance_sq) {
                        continue;
                    }

                    if (used_b[static_cast<size_t>(b_idx)]) {
                        continue;
                    }

                    cv::Mat query_b = points_b.row(b_idx);
                    index_a.knnSearch(query_b, reverse_index, reverse_dist, 1, search_params);

                    if (reverse_index[0] != static_cast<int>(a_idx) || reverse_dist[0] > tolerance_sq) {
                        continue;
                    }

                    used_b[static_cast<size_t>(b_idx)] = 1;
                    matches.emplace_back(a_idx, static_cast<size_t>(b_idx));
                }

                if (matches.empty()) {
                    LOG_WARNING("⚠️  " + scene + "/" + image + ": no mutually close keypoints within " + std::to_string(tolerance_px) + "px");
                    continue;
                }

                total_pairs += matches.size();

                // Pure intersection: just insert spatially matched keypoints with native parameters

                // Insert spatially matched keypoints with their native detector parameters
                for (const auto& match : matches) {
                    const auto& record_a = records_a[match.first];
                    const auto& record_b = records_b[match.second];

                    // Insert keypoint A with its native parameters from source set A
                    int new_a_id = db.insertLockedKeypoint(output_a_id, scene, image, record_a.keypoint, true);
                    if (new_a_id >= 0) {
                        ++total_inserted_a;
                    } else {
                        LOG_ERROR("❌ Failed to insert keypoint for " + scene + "/" + image + " into " + output_a_name);
                    }

                    // Insert keypoint B with its native parameters from source set B
                    int new_b_id = db.insertLockedKeypoint(output_b_id, scene, image, record_b.keypoint, true);
                    if (new_b_id >= 0) {
                        ++total_inserted_b;
                    } else {
                        LOG_ERROR("❌ Failed to insert keypoint for " + scene + "/" + image + " into " + output_b_name);
                    }
                }

                // Pure intersection sets use native keypoint parameters - no attribute copying needed
            }
        }

        LOG_INFO("🎯 Intersection complete within " + std::to_string(tolerance_px) + "px");
        LOG_INFO("📊 Matched " + std::to_string(total_pairs) + " keypoint pairs across " + std::to_string(processed_images) + " images");
        LOG_INFO("🅰️  " + output_a_name + ": " + std::to_string(total_inserted_a) + " keypoints inserted from " + source_a_name +
                 " (candidates: " + std::to_string(total_candidates_a) + ")");
        LOG_INFO("🅱️  " + output_b_name + ": " + std::to_string(total_inserted_b) + " keypoints inserted from " + source_b_name +
                 " (candidates: " + std::to_string(total_candidates_b) + ")");

        if (skipped_images > 0) {
            LOG_WARNING("⚠️  Skipped " + std::to_string(skipped_images) + " images due to missing data or zero matches");
        }

        LOG_INFO("✅ Pure intersection sets created - keypoints retain native detector parameters");

    } else if (command == "generate-non-overlapping") {
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
            // Parse detector type
            KeypointGenerator detector_type = KeypointGeneratorFactory::parseDetectorType(detector_str);
            
            LOG_INFO("🔍 Generating non-overlapping keypoints using " + detector_str + " detector");
            LOG_INFO("📁 Data folder: " + data_folder);
            LOG_INFO("📏 Minimum distance: " + std::to_string(min_distance) + "px");
            LOG_INFO("📝 Keypoint set name: " + set_name);
            
            namespace fs = boost::filesystem;
            if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
                std::cerr << "❌ Data folder does not exist: " + data_folder << std::endl;
                return 1;
            }

            // Create keypoint set with overlap constraint tracking
            int set_id = db.createKeypointSetWithOverlap(
                set_name,
                detector_str, 
                "non_overlapping_detection",
                2000,
                data_folder,
                detector_str + " detector with non-overlapping constraint (min_distance=" + std::to_string(min_distance) + "px)",
                40,
                true,  // overlap_filtering = true
                min_distance
            );
            
            if (set_id == -1) {
                std::cerr << "❌ Failed to create keypoint set: " + set_name << std::endl;
                return 1;
            }
            
            LOG_INFO("✅ Created keypoint set with ID: " + std::to_string(set_id));
            
            // Create detector with non-overlapping constraint
            auto detector = KeypointGeneratorFactory::create(detector_type, true, min_distance);
            KeypointParams params;
            params.max_features = 2000;
            
            int total_keypoints = 0;

            // Process each scene
            for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
                if (!fs::is_directory(scene_entry)) continue;
                
                std::string scene_name = scene_entry.path().filename().string();
                LOG_INFO("📁 Processing scene: " + scene_name);
                
                // Process each image independently (1.ppm to 6.ppm)
                for (int i = 1; i <= 6; ++i) {
                    fs::path image_path = scene_entry.path() / (std::to_string(i) + ".ppm");
                    if (!fs::exists(image_path)) {
                        std::cerr << "❌ Image not found: " << image_path.string() << std::endl;
                        continue;
                    }
                    
                    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
                    if (image.empty()) {
                        std::cerr << "❌ Could not load image: " << image_path.string() << std::endl;
                        continue;
                    }
                    
                    // Detect non-overlapping keypoints
                    std::vector<cv::KeyPoint> keypoints = detector->detectNonOverlapping(image, min_distance, params);
                    
                    // Store keypoints for this image
                    std::string image_name = std::to_string(i) + ".ppm";
                    if (db.storeLockedKeypointsForSet(set_id, scene_name, image_name, keypoints)) {
                        total_keypoints += keypoints.size();
                        LOG_INFO("  ✅ " + scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("  ❌ Failed to store keypoints for " + scene_name + "/" + image_name);
                    }
                }
            }
            
            LOG_INFO("🎉 Generation complete! Non-overlapping " + detector_str + " keypoints stored in set: " + set_name);
            LOG_INFO("📊 Total keypoints generated: " + std::to_string(total_keypoints));
            
        } catch (const std::exception& e) {
            LOG_ERROR("❌ Error: " + std::string(e.what()));
            return 1;
        }

    } else if (command == "list-detectors") {
        auto detectors = KeypointGeneratorFactory::getSupportedDetectors();
        std::cout << "🔧 Supported detectors (" << detectors.size() << "):" << std::endl;
        for (const auto& detector : detectors) {
            float recommended_distance = KeypointGeneratorFactory::getRecommendedMinDistance(
                KeypointGeneratorFactory::parseDetectorType(detector), 32);
            std::cout << "  📍 " << detector << " (recommended min_distance for 32px patches: " 
                      << recommended_distance << "px)" << std::endl;
        }

    } else {
        std::cerr << "❌ Unknown command: " << command << std::endl;
        std::cerr << "Run without arguments to see available commands." << std::endl;
        return 1;
    }

    return 0;
}
