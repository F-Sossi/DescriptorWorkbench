#include "generation_detector.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include "src/core/utils/PythonEnvironment.hpp"
#include "thesis_project/logging.hpp"
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace thesis_project::cli::keypoint_commands {

int generateKorniaKeynet(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " generate-kornia-keynet <data_folder> [set_name] [--max-features N] [--device cpu|cuda|auto] [--mode independent|projected] [--overwrite]" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-kornia-keynet ../data keynet_reference --max-features 8000 --device auto --overwrite" << std::endl;
        return 1;
    }

    std::string data_folder = argv[2];
    int arg_index = 3;

    std::string set_name;
    if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
        set_name = argv[arg_index];
        ++arg_index;
    } else {
        set_name = "keynet_kornia_" + std::to_string(std::time(nullptr));
    }

    int max_kp = 2000;
    std::string device_arg = "auto";
    std::string mode_arg = "independent";
    bool overwrite = false;

    while (arg_index < argc) {
        std::string arg = argv[arg_index];
        if (arg == "--max-features") {
            if (arg_index + 1 >= argc) {
                std::cerr << "Missing value for --max-features" << std::endl;
                return 1;
            }
            max_kp = std::stoi(argv[arg_index + 1]);
            arg_index += 2;
        } else if (arg == "--device") {
            if (arg_index + 1 >= argc) {
                std::cerr << "Missing value for --device" << std::endl;
                return 1;
            }
            device_arg = argv[arg_index + 1];
            arg_index += 2;
        } else if (arg == "--mode") {
            if (arg_index + 1 >= argc) {
                std::cerr << "Missing value for --mode" << std::endl;
                return 1;
            }
            mode_arg = argv[arg_index + 1];
            arg_index += 2;
        } else if (arg == "--overwrite") {
            overwrite = true;
            ++arg_index;
        } else {
            std::cerr << "Unknown option for generate-kornia-keynet: " << arg << std::endl;
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

    using namespace thesis_project::utils;
    PythonEnvironment::Requirements reqs;
    reqs.packages = {"kornia", "torch", "cv2", "sqlite3"};

    auto env_info = PythonEnvironment::detect(reqs);
    if (!env_info.is_valid()) {
        LOG_ERROR("No valid Python environment with required packages (kornia, torch, cv2, sqlite3)");
        if (!env_info.missing_packages.empty()) {
            std::ostringstream msg;
            msg << "Missing packages: ";
            for (size_t i = 0; i < env_info.missing_packages.size(); ++i) {
                if (i > 0) msg << ", ";
                msg << env_info.missing_packages[i];
            }
            LOG_ERROR(msg.str());
        }
        return 1;
    }

    std::filesystem::path script_path = std::filesystem::current_path().parent_path() / "scripts" / "generate_keynet_keypoints.py";
    if (!std::filesystem::exists(script_path)) {
        script_path = std::filesystem::path("../scripts/generate_keynet_keypoints.py");
        if (!std::filesystem::exists(script_path)) {
            LOG_ERROR("Could not find generate_keynet_keypoints.py script");
            return 1;
        }
    }

    std::vector<std::string> args;
    args.push_back("--data_dir \"" + data_folder + "\"");
    args.push_back("--db_path experiments.db");
    args.push_back("--max_keypoints " + std::to_string(max_kp));
    args.push_back("--device " + device_arg);
    args.push_back("--mode " + mode_arg);
    args.push_back("--set-name \"" + set_name + "\"");
    if (overwrite) {
        args.push_back("--overwrite");
    }

    std::string cmd = PythonEnvironment::generateCommand(script_path.string(), args, env_info);

    if (cmd.empty()) {
        LOG_ERROR("Failed to generate Python command");
        return 1;
    }

    LOG_INFO("Using Python environment: " + PythonEnvironment::typeToString(env_info.type) +
             " (" + env_info.environment_name + ")");
    LOG_INFO("Executing: " + cmd);

    int result = std::system(cmd.c_str());
    if (result != 0) {
        LOG_ERROR("Kornia KeyNet generation failed. Exit code: " + std::to_string(result));
        return 1;
    }

    LOG_INFO("Kornia KeyNet generation completed successfully for set: " + set_name);
    return 0;
}

int generateDetector(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
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
                std::cerr << "Missing value for --max-features" << std::endl;
                return 1;
            }
            max_features = std::stoi(argv[arg_index++]);
        } else if (arg == "--overwrite") {
            overwrite = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    try {
        KeypointGenerator detector_type = KeypointGeneratorFactory::parseDetectorType(detector_str);

        namespace fs = boost::filesystem;
        if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
            std::cerr << "Data folder does not exist: " << data_folder << std::endl;
            return 1;
        }

        LOG_INFO("Generating keypoints using " + detector_str + " detector");
        LOG_INFO("Data folder: " + data_folder);
        LOG_INFO("Keypoint set name: " + set_name);
        LOG_INFO("Max features per image: " + std::to_string(max_features > 0 ? max_features : -1));

        int set_id = db.getKeypointSetId(set_name);
        if (set_id >= 0) {
            if (!overwrite) {
                std::cerr << "Keypoint set already exists: " << set_name << " (use --overwrite to replace)" << std::endl;
                return 1;
            }

            if (db.clearAllDetectorAttributesForSet(set_id)) {
                LOG_INFO("database clear all detector attributes for set: " + set_name);
            } else {
                LOG_INFO("Database clear all detector attributes for set: " + set_name + " failed");
            }

            if (db.clearKeypointsForSet(set_id)) {
                LOG_INFO("Keypoint clear all keypoints for set: " + set_name);
            } else {
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
                std::cerr << "Failed to create keypoint set: " + set_name << std::endl;
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
        int images_processed = 0;

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

                std::vector<cv::KeyPoint> keypoints = detector->detect(image, params);
                trimByResponse(keypoints, max_features);

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

        LOG_INFO("Generation complete! " + detector_str + " keypoints stored in set: " + set_name);
        LOG_INFO("Total keypoints generated: " + std::to_string(total_keypoints));

    } catch (const std::exception& e) {
        LOG_ERROR("Error generating keypoints: " + std::string(e.what()));
        return 1;
    }

    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
