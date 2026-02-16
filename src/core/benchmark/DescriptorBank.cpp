#include "DescriptorBank.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace thesis_project::benchmark {

DescriptorBank::DescriptorBank(const PatchScope& scope)
    : scope_(scope) {
    // Initialize all scope keys as missing
    for (const auto& key : scope_) {
        missing_.insert(key);
    }
}

size_t DescriptorBank::loadFromDatabase(database::DatabaseManager& db, int cache_id) {
    size_t loaded = 0;

    for (const auto& key : scope_) {
        // Try to load from database
        auto desc = db.loadPatchBenchmarkDescriptor(
            cache_id, key.scene, key.difficulty, key.target);

        if (desc && !desc->empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            descriptors_[key] = std::move(*desc);
            missing_.erase(key);
            loaded++;
        }
    }

    return loaded;
}

void DescriptorBank::extractMissing(
    const std::string& patches_dir,
    bool color,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    const ProgressCallback& progress) {

    // Group missing keys by scene for efficient loading
    std::unordered_map<std::string, std::vector<PatchKey>> by_scene_map;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& key : missing_) {
            by_scene_map[key.scene].push_back(key);
        }
    }

    if (by_scene_map.empty()) {
        return;  // Nothing to extract
    }

    // Convert to vector for OpenMP indexed access
    std::vector<std::pair<std::string, std::vector<PatchKey>>> by_scene(
        by_scene_map.begin(), by_scene_map.end());

    const int total = static_cast<int>(by_scene.size());
    std::atomic<int> completed{0};

#ifdef _OPENMP
    #pragma omp parallel
    {
        // Each thread gets its own extractor clone
        auto thread_extractor = extractor.clone();

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total; ++i) {
            const auto& [scene_name, keys] = by_scene[i];

            // Construct scene directory path
            std::string scene_dir = (std::filesystem::path(patches_dir) / scene_name).string();

            if (!std::filesystem::exists(scene_dir)) {
                #pragma omp critical
                std::cerr << "Warning: Scene directory not found: " << scene_dir << std::endl;
                completed++;
                continue;
            }

            // Load the scene patches
            patches::PatchLoader::ScenePatches scene;
            try {
                scene = patches::PatchLoader::loadScene(scene_dir, color);
            } catch (const std::exception& e) {
                #pragma omp critical
                std::cerr << "Warning: Failed to load scene " << scene_dir << ": " << e.what() << std::endl;
                completed++;
                continue;
            }

            // Extract descriptors for each key in this scene
            for (const auto& key : keys) {
                std::vector<cv::Mat> patches_to_extract;

                // Get the patches for this key
                if (key.target == "ref") {
                    patches_to_extract = scene.ref.patches;
                } else if (key.difficulty == "easy") {
                    auto it = scene.easy.find(key.target);
                    if (it != scene.easy.end()) {
                        patches_to_extract = it->second.patches;
                    }
                } else if (key.difficulty == "hard") {
                    auto it = scene.hard.find(key.target);
                    if (it != scene.hard.end()) {
                        patches_to_extract = it->second.patches;
                    }
                } else if (key.difficulty == "tough") {
                    auto it = scene.tough.find(key.target);
                    if (it != scene.tough.end()) {
                        patches_to_extract = it->second.patches;
                    }
                }

                if (patches_to_extract.empty()) {
                    continue;  // No patches for this key
                }

                // Resize patches if needed (e.g., 65x65 -> 32x32 for CNN)
                if (thread_extractor->requiresResize()) {
                    patches_to_extract = patches::PatchLoader::resizeForCNN(
                        patches_to_extract, thread_extractor->expectedPatchSize());
                }

                // Extract descriptors
                cv::Mat desc = thread_extractor->extractFromPatches(patches_to_extract, params);

                if (!desc.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    descriptors_[key] = desc;
                    missing_.erase(key);
                }
            }

            int done = ++completed;
            if (progress) {
                #pragma omp critical
                progress(done, total, "Extracting " + scene_name);
            }
        }
    }
#else
    // Non-OpenMP fallback: sequential processing
    for (int i = 0; i < total; ++i) {
        const auto& [scene_name, keys] = by_scene[i];

        int done = ++completed;
        if (progress) {
            progress(done, total, "Extracting " + scene_name);
        }

        std::string scene_dir = (std::filesystem::path(patches_dir) / scene_name).string();

        if (!std::filesystem::exists(scene_dir)) {
            std::cerr << "Warning: Scene directory not found: " << scene_dir << std::endl;
            continue;
        }

        patches::PatchLoader::ScenePatches scene;
        try {
            scene = patches::PatchLoader::loadScene(scene_dir, color);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load scene " << scene_dir << ": " << e.what() << std::endl;
            continue;
        }

        for (const auto& key : keys) {
            std::vector<cv::Mat> patches_to_extract;

            if (key.target == "ref") {
                patches_to_extract = scene.ref.patches;
            } else if (key.difficulty == "easy") {
                auto it = scene.easy.find(key.target);
                if (it != scene.easy.end()) {
                    patches_to_extract = it->second.patches;
                }
            } else if (key.difficulty == "hard") {
                auto it = scene.hard.find(key.target);
                if (it != scene.hard.end()) {
                    patches_to_extract = it->second.patches;
                }
            } else if (key.difficulty == "tough") {
                auto it = scene.tough.find(key.target);
                if (it != scene.tough.end()) {
                    patches_to_extract = it->second.patches;
                }
            }

            if (patches_to_extract.empty()) {
                continue;
            }

            if (extractor.requiresResize()) {
                patches_to_extract = patches::PatchLoader::resizeForCNN(
                    patches_to_extract, extractor.expectedPatchSize());
            }

            cv::Mat desc = extractor.extractFromPatches(patches_to_extract, params);

            if (!desc.empty()) {
                std::lock_guard<std::mutex> lock(mutex_);
                descriptors_[key] = desc;
                missing_.erase(key);
            }
        }
    }
#endif
}

bool DescriptorBank::storeToDatabase(database::DatabaseManager& db, int cache_id) {
    bool success = true;

    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [key, desc] : descriptors_) {
        if (!db.storePatchBenchmarkDescriptor(cache_id, key.scene, key.difficulty, key.target, desc)) {
            std::cerr << "Warning: Failed to store descriptor for " << key.toString() << std::endl;
            success = false;
        }
    }

    return success;
}

const cv::Mat& DescriptorBank::get(const PatchKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = descriptors_.find(key);
    if (it != descriptors_.end()) {
        return it->second;
    }
    return empty_mat_;
}

const cv::Mat& DescriptorBank::get(
    const std::string& scene,
    const std::string& difficulty,
    const std::string& target) const {

    return get(PatchKey{scene, difficulty, target});
}

cv::Mat DescriptorBank::getRow(const PatchKey& key, int idx) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = descriptors_.find(key);
    if (it == descriptors_.end() || it->second.empty()) {
        return {};
    }

    if (idx < 0 || idx >= it->second.rows) {
        return {};
    }

    return it->second.row(idx).clone();
}

int DescriptorBank::descriptorDimension() const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [key, desc] : descriptors_) {
        if (!desc.empty()) {
            return desc.cols;
        }
    }
    return 0;
}

int DescriptorBank::totalPatches() const {
    std::lock_guard<std::mutex> lock(mutex_);

    int total = 0;
    for (const auto& [key, desc] : descriptors_) {
        total += desc.rows;
    }
    return total;
}

std::unordered_map<std::string, std::vector<PatchKey>> DescriptorBank::groupByScene() const {
    std::unordered_map<std::string, std::vector<PatchKey>> result;
    for (const auto& key : scope_) {
        result[key.scene].push_back(key);
    }
    return result;
}

} // namespace thesis_project::benchmark
