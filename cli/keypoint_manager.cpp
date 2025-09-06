#include "thesis_project/database/DatabaseManager.hpp"
#include "src/core/keypoints/locked_in_keypoints.hpp"
#include "src/core/processing/processor_utils.hpp"
#include "thesis_project/logging.hpp"
#include <iostream>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <fstream>
#include <ctime>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

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
        std::cout << "  Import/Export:" << std::endl;
        std::cout << "    import-csv <csv_folder> [set_name]        - Import keypoints from CSV files" << std::endl;
        std::cout << "    export-csv <output_folder> [set_id]       - Export keypoints from DB to CSV" << std::endl;
        std::cout << "  Information:" << std::endl;
        std::cout << "    list-sets                                 - List all available keypoint sets" << std::endl;
        std::cout << "    list-scenes [set_id]                      - List scenes in database (optionally for specific set)" << std::endl;
        std::cout << "    count <scene> <image> [set_id]            - Count keypoints for specific scene/image" << std::endl;
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
        std::cerr << "⚠️  Warning: Failed to apply database optimizations" << std::endl;
    }

    if (command == "import-csv") {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " import-csv <csv_folder>" << std::endl;
            std::cerr << "  Example: " << argv[0] << " import-csv ../reference_keypoints" << std::endl;
            return 1;
        }

        std::string csv_folder = argv[2];
        LOG_INFO("🔄 Importing keypoints from CSV folder: " + csv_folder);

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
            LOG_INFO("📁 Processing scene: " + scene_name);
            
            // Iterate through each CSV file in the scene folder
            for (const auto& csv_entry : fs::directory_iterator(scene_entry)) {
                if (csv_entry.path().extension() != ".csv") continue;
                
                std::string csv_file = csv_entry.path().string();
                std::string image_name = csv_entry.path().stem().string() + ".ppm";
                
                try {
                    std::vector<cv::KeyPoint> keypoints = LockedInKeypoints::readKeypointsFromCSV(csv_file);
                    
                    if (db.storeLockedKeypoints(scene_name, image_name, keypoints)) {
                        total_imported += keypoints.size();
                        LOG_INFO("  ✅ " + scene_name + "/" + image_name + ": " + std::to_string(keypoints.size()) + " keypoints");
                    } else {
                        LOG_ERROR("  ❌ Failed to store: " + scene_name + "/" + image_name);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("  ❌ Error processing " + csv_file + ": " + std::string(e.what()));
                }
            }
        }
        
        LOG_INFO("🎉 Import complete! Total keypoints imported: " + std::to_string(total_imported));

    } else if (command == "generate-projected") {
        if (argc < 3 || argc > 4) {
            std::cerr << "Usage: " << argv[0] << " generate-projected <data_folder> [name]" << std::endl;
            std::cerr << "  Example: " << argv[0] << " generate-projected ../data" << std::endl;
            std::cerr << "  Example: " << argv[0] << " generate-projected ../data my_projected_keypoints" << std::endl;
            return 1;
        }
        
        std::string data_folder = argv[2];
        std::string set_name = (argc == 4) ? argv[3] : "homography_projection_" + std::to_string(std::time(nullptr));
        
        LOG_INFO("🔄 Generating homography projected keypoints from: " + data_folder);
        LOG_INFO("📝 Keypoint set name: " + set_name);
        
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
        
        LOG_INFO("✅ Created keypoint set with ID: " + std::to_string(set_id));
        
        // Generate keypoints using homography projection method
        LOG_INFO("🔍 Generating keypoints with homography projection and boundary filtering...");
        
        try {
            // Use the existing LockedInKeypoints method but store to specific set
            LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db, set_id);
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
        
        LOG_INFO("🔄 Generating independent detection keypoints from: " + data_folder);
        LOG_INFO("📝 Keypoint set name: " + set_name);
        
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
        
        LOG_INFO("✅ Created keypoint set with ID: " + std::to_string(set_id));
        
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
            
            LOG_INFO("🎉 Generation complete! Independent detection keypoints stored in set: " + set_name);
            LOG_INFO("📊 Total keypoints generated: " + std::to_string(total_keypoints));
            
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
        LOG_INFO("🔄 Generating fresh locked keypoints from: " + data_folder);
        
        namespace fs = boost::filesystem;
        if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
            std::cerr << "❌ Data folder does not exist: " << data_folder << std::endl;
            return 1;
        }

        // Clear existing keypoints for all scenes in the data folder
        LOG_INFO("🗑️  Clearing existing keypoints from database...");
        for (const auto& scene_entry : fs::directory_iterator(data_folder)) {
            if (!fs::is_directory(scene_entry)) continue;
            std::string scene_name = scene_entry.path().filename().string();
            if (!db.clearSceneKeypoints(scene_name)) {
                LOG_ERROR("❌ Failed to clear keypoints for scene: " + scene_name);
                return 1;
            }
        }
        
        // Generate fresh keypoints using tested boundary-filtering logic
        LOG_INFO("🔍 Generating new locked keypoints with proper boundary filtering...");
        
        try {
            LockedInKeypoints::generateLockedInKeypointsToDatabase(data_folder, db);
            LOG_INFO("🎉 Generation complete! Keypoints generated with 40px boundary filtering.");
            
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
        LOG_INFO("💾 Exporting keypoints to CSV folder: " + output_folder);
        
        // Create output directory
        std::filesystem::create_directories(output_folder);
        
        auto scenes = db.getAvailableScenes();
        if (scenes.empty()) {
            LOG_INFO("ℹ️  No keypoints found in database to export");
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
                    LOG_INFO("  ✅ " + scene + "/" + csv_filename + ": " + std::to_string(keypoints.size()) + " keypoints");
                }
            }
        }
        
        LOG_INFO("🎉 Export complete! Total keypoints exported: " + std::to_string(total_exported));

    } else if (command == "list-sets") {
        auto sets = db.getAvailableKeypointSets();
        std::cout << "📋 Available keypoint sets (" << sets.size() << "):" << std::endl;
        for (const auto& [id, name, method] : sets) {
            // Count keypoints in this set
            std::string sql = "SELECT COUNT(*) FROM locked_keypoints WHERE keypoint_set_id = " + std::to_string(id);
            // For now, let's show the basic info - we can add counts later if needed
            std::cout << "  🔧 ID " << id << ": " << name << " (" << method << ")" << std::endl;
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

    } else {
        std::cerr << "❌ Unknown command: " << command << std::endl;
        std::cerr << "Run without arguments to see available commands." << std::endl;
        return 1;
    }

    return 0;
}
