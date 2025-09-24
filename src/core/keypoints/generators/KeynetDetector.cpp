#include "KeynetDetector.hpp"
#include "thesis_project/logging.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <cstdlib>

namespace thesis_project {

KeynetDetector::KeynetDetector(int max_keypoints, const std::string& device)
    : max_keypoints_(max_keypoints), device_(device) {

    // Find the Python script path
    // First check in scripts/ directory relative to current working directory
    const std::vector<std::string> script_filenames = {
        "generate_keynet_keypoints.py",
        "keynet_single_image.py"
    };

    std::filesystem::path current_dir = std::filesystem::current_path();

    for (const auto& script_name : script_filenames) {
        std::filesystem::path search_dir = current_dir;
        for (int depth = 0; depth < 8; ++depth) {
            std::filesystem::path candidate = search_dir / "scripts" / script_name;
            if (std::filesystem::exists(candidate)) {
                python_script_path_ = candidate.string();
                break;
            }
            if (search_dir.has_parent_path()) {
                search_dir = search_dir.parent_path();
            }
        }
        if (!python_script_path_.empty()) {
            break;
        }
    }

    if (python_script_path_.empty()) {
        LOG_WARNING("KeyNet Python script not found in any parent directories");
        python_script_path_ = "scripts/generate_keynet_keypoints.py"; // Fallback relative path
    } else {
        LOG_INFO("KeyNet detector initialized with script: " + python_script_path_);
    }
}

std::vector<cv::KeyPoint> KeynetDetector::detect(
    const cv::Mat& image,
    const KeypointParams& params
) {
    if (image.empty()) {
        LOG_ERROR("KeyNet detector: Input image is empty");
        return {};
    }

    // Use max_features from params if provided, otherwise use constructor value
    int target_keypoints = (params.max_features > 0) ? params.max_features : max_keypoints_;

    try {
        // Generate temporary file paths
        std::string temp_image = generateTempFilename(".png");
        std::string temp_csv = generateTempFilename(".csv");

        // Save image to temporary file
        saveImageToTemp(image, temp_image);

        // Execute Python KeyNet detection
        bool success = executePythonKeyNet(temp_image, temp_csv);

        std::vector<cv::KeyPoint> keypoints;
        if (success) {
            // Load keypoints from CSV
            keypoints = loadKeypointsFromTemp(temp_csv);

            // Limit to requested number of keypoints (keep highest response)
            if (keypoints.size() > static_cast<size_t>(target_keypoints)) {
                std::partial_sort(keypoints.begin(),
                                keypoints.begin() + target_keypoints,
                                keypoints.end(),
                                [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                                    return a.response > b.response;
                                });
                keypoints.resize(target_keypoints);
            }

            LOG_INFO("KeyNet detector: Generated " + std::to_string(keypoints.size()) + " keypoints");
        } else {
            LOG_ERROR("KeyNet detector: Python execution failed");
        }

        // Cleanup temporary files
        std::filesystem::remove(temp_image);
        std::filesystem::remove(temp_csv);

        return keypoints;

    } catch (const std::exception& e) {
        LOG_ERROR("KeyNet detector error: " + std::string(e.what()));
        return {};
    }
}

std::vector<cv::KeyPoint> KeynetDetector::detectNonOverlapping(
    const cv::Mat& image,
    float min_distance,
    const KeypointParams& params
) {
    // For now, just call regular detect and let NonOverlappingKeypointGenerator wrapper handle it
    // This could be optimized later to do spatial filtering in Python
    return detect(image, params);
}

bool KeynetDetector::isAvailable() {
    // Check if Python and Kornia are available by running a simple test
    std::string test_command = "python3 -c \"import kornia.feature as KF; print('OK')\" 2>/dev/null";
    int result = std::system(test_command.c_str());
    return (result == 0);
}

void KeynetDetector::saveImageToTemp(const cv::Mat& image, const std::string& temp_path) const {
    if (!cv::imwrite(temp_path, image)) {
        throw std::runtime_error("Failed to save temporary image: " + temp_path);
    }
}

std::vector<cv::KeyPoint> KeynetDetector::loadKeypointsFromTemp(const std::string& csv_path) const {
    std::vector<cv::KeyPoint> keypoints;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open keypoint CSV: " + csv_path);
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        // Skip header line
        if (first_line) {
            first_line = false;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> values;

        // Parse CSV line
        while (std::getline(ss, cell, ',')) {
            values.push_back(cell);
        }

        // Expected format: x,y,size,angle,response
        if (values.size() >= 5) {
            try {
                float x = std::stof(values[0]);
                float y = std::stof(values[1]);
                float size = std::stof(values[2]);
                float angle = std::stof(values[3]);
                float response = std::stof(values[4]);

                cv::KeyPoint kp(x, y, size, angle, response);
                keypoints.push_back(kp);
            } catch (const std::exception& e) {
                LOG_WARNING("KeyNet: Skipping invalid keypoint data: " + line);
                continue;
            }
        }
    }

    return keypoints;
}

bool KeynetDetector::executePythonKeyNet(const std::string& image_path,
                                       const std::string& output_csv_path) const {
    // Build Python command with conda environment activation
    // TODO: MAJOR - Make this portable for other systems! Currently hardcoded to Frank's environment
    // Need to detect conda path and environment name automatically or via config
    std::ostringstream cmd;
    cmd << "/bin/bash -c \"source /home/frank/miniforge3/etc/profile.d/conda.sh && conda activate descriptor-compare && "
        << "python3 \"" << python_script_path_ << "\""
        << " --input \"" << image_path << "\""
        << " --output \"" << output_csv_path << "\""
        << " --max_keypoints " << max_keypoints_
        << " --device " << device_ << "\"";
        // << " 2>/dev/null"; // Temporarily enable stderr for debugging

    LOG_INFO("Executing KeyNet command: " + cmd.str());

    int result = std::system(cmd.str().c_str());
    LOG_INFO("KeyNet command result: " + std::to_string(result));

    // Check if output file was created
    if (std::filesystem::exists(output_csv_path)) {
        std::ifstream file(output_csv_path);
        std::string line;
        int line_count = 0;
        while (std::getline(file, line)) {
            line_count++;
        }
        LOG_INFO("Output CSV has " + std::to_string(line_count) + " lines");
    } else {
        LOG_ERROR("Output CSV file was not created: " + output_csv_path);
    }

    return (result == 0);
}

std::string KeynetDetector::generateTempFilename(const std::string& extension) const {
    // Generate unique filename using timestamp and random number
    auto now = std::chrono::high_resolution_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);

    std::string filename = "/tmp/keynet_" + std::to_string(timestamp) + "_" +
                          std::to_string(dis(gen)) + extension;

    return filename;
}

} // namespace thesis_project
