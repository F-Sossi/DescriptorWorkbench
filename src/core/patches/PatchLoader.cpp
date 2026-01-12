#include "PatchLoader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <algorithm>

namespace thesis_project {
namespace patches {

PatchLoader::PatchSet PatchLoader::loadStackedPNG(const std::string& png_path, bool color) {
    PatchSet result;

    // Extract name from path (e.g., "e1" from "/path/to/scene/e1.png")
    std::filesystem::path p(png_path);
    result.name = p.stem().string();

    // Also extract scene name from parent directory
    if (p.has_parent_path()) {
        result.scene_name = p.parent_path().filename().string();
    }

    // Load the stacked PNG
    cv::Mat stacked_img = cv::imread(png_path, color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (stacked_img.empty()) {
        throw std::runtime_error("Failed to load patch file: " + png_path);
    }

    // Validate dimensions
    if (stacked_img.cols != PATCH_SIZE) {
        throw std::runtime_error("Invalid patch width: expected " +
                                 std::to_string(PATCH_SIZE) + ", got " +
                                 std::to_string(stacked_img.cols) +
                                 " in " + png_path);
    }

    if (stacked_img.rows % PATCH_SIZE != 0) {
        throw std::runtime_error("Invalid patch height: " +
                                 std::to_string(stacked_img.rows) +
                                 " is not divisible by " +
                                 std::to_string(PATCH_SIZE) +
                                 " in " + png_path);
    }

    result.patches = extractPatches(stacked_img);
    return result;
}

std::vector<cv::Mat> PatchLoader::extractPatches(const cv::Mat& stacked_img) {
    const int num_patches = stacked_img.rows / PATCH_SIZE;
    std::vector<cv::Mat> patches;
    patches.reserve(num_patches);

    for (int i = 0; i < num_patches; ++i) {
        cv::Rect roi(0, i * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE);
        // Clone to ensure each patch owns its data
        patches.push_back(stacked_img(roi).clone());
    }

    return patches;
}

PatchLoader::ScenePatches PatchLoader::loadScene(const std::string& scene_dir, bool color) {
    ScenePatches scene;

    std::filesystem::path dir(scene_dir);
    scene.scene_name = dir.filename().string();

    // Load reference patches
    std::string ref_path = (dir / "ref.png").string();
    if (std::filesystem::exists(ref_path)) {
        scene.ref = loadStackedPNG(ref_path, color);
    } else {
        throw std::runtime_error("Reference patches not found: " + ref_path);
    }

    // Load easy patches (e1-e5)
    for (int i = 1; i <= 5; ++i) {
        std::string key = "e" + std::to_string(i);
        std::string path = (dir / (key + ".png")).string();
        if (std::filesystem::exists(path)) {
            scene.easy[key] = loadStackedPNG(path, color);
        }
    }

    // Load hard patches (h1-h5)
    for (int i = 1; i <= 5; ++i) {
        std::string key = "h" + std::to_string(i);
        std::string path = (dir / (key + ".png")).string();
        if (std::filesystem::exists(path)) {
            scene.hard[key] = loadStackedPNG(path, color);
        }
    }

    // Load tough patches (t1-t5) - may not exist in all versions
    for (int i = 1; i <= 5; ++i) {
        std::string key = "t" + std::to_string(i);
        std::string path = (dir / (key + ".png")).string();
        if (std::filesystem::exists(path)) {
            scene.tough[key] = loadStackedPNG(path, color);
        }
    }

    return scene;
}

cv::Mat PatchLoader::resizeForCNN(const cv::Mat& patch65, int target_size) {
    if (patch65.empty()) {
        return cv::Mat();
    }

    cv::Mat resized;
    // Use INTER_AREA for downsampling (anti-aliasing) - matches HPatches protocol
    cv::resize(patch65, resized, cv::Size(target_size, target_size),
               0, 0, cv::INTER_AREA);
    return resized;
}

std::vector<cv::Mat> PatchLoader::resizeForCNN(const std::vector<cv::Mat>& patches65,
                                                int target_size) {
    std::vector<cv::Mat> resized;
    resized.reserve(patches65.size());

    for (const auto& patch : patches65) {
        resized.push_back(resizeForCNN(patch, target_size));
    }

    return resized;
}

int PatchLoader::countPatches(const std::string& png_path) {
    // Read only the header to get dimensions without loading full image
    cv::Mat img = cv::imread(png_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        return 0;
    }
    return img.rows / PATCH_SIZE;
}

std::vector<std::string> PatchLoader::listScenes(const std::string& dataset_path) {
    std::vector<std::string> scenes;

    if (!std::filesystem::exists(dataset_path)) {
        throw std::runtime_error("Dataset path does not exist: " + dataset_path);
    }

    for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            // Only include i_* and v_* directories
            if (name.size() > 2 && (name[0] == 'i' || name[0] == 'v') && name[1] == '_') {
                scenes.push_back(entry.path().string());
            }
        }
    }

    // Sort for consistent ordering
    std::sort(scenes.begin(), scenes.end());

    return scenes;
}

bool PatchLoader::isIlluminationScene(const std::string& scene_name) {
    std::filesystem::path p(scene_name);
    std::string name = p.filename().string();
    return name.size() > 2 && name[0] == 'i' && name[1] == '_';
}

bool PatchLoader::isViewpointScene(const std::string& scene_name) {
    std::filesystem::path p(scene_name);
    std::string name = p.filename().string();
    return name.size() > 2 && name[0] == 'v' && name[1] == '_';
}

} // namespace patches
} // namespace thesis_project
