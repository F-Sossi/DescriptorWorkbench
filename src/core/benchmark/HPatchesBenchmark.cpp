#include "HPatchesBenchmark.hpp"
#include "core/patches/PatchLoader.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace thesis_project::benchmark {

using namespace patches;

HPatchesBenchmark::Results HPatchesBenchmark::run(
    const Config& config,
    IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    ProgressCallback progress_callback) {

    Results results;
    results.descriptor_name = extractor.name();
    results.descriptor_dimension = extractor.descriptorSize();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get list of scenes
    std::vector<std::string> scene_dirs;
    if (config.scenes.empty()) {
        scene_dirs = PatchLoader::listScenes(config.patches_dir);
    } else {
        scene_dirs = config.scenes;
    }

    results.num_scenes = static_cast<int>(scene_dirs.size());

    if (config.verbose) {
        std::cout << "HPatches Benchmark: " << results.descriptor_name
                  << " (" << results.descriptor_dimension << "D)" << std::endl;
        std::cout << "Processing " << scene_dirs.size() << " scenes..." << std::endl;
    }

    // Accumulators for different categories
    Accumulator overall, easy_all, hard_all, tough_all;
    Accumulator illumination_all, viewpoint_all;
    Accumulator illumination_easy, illumination_hard;
    Accumulator viewpoint_easy, viewpoint_hard;

    int total_patches = 0;
    int scenes_processed = 0;

    const bool allow_parallel = (config.num_threads != 1);

#ifdef _OPENMP
    if (allow_parallel && config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }
#endif

    if (allow_parallel) {
        std::atomic<int> completed{0};

        #pragma omp parallel
        {
            auto local_extractor = extractor.clone();

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < scene_dirs.size(); ++i) {
                const auto& scene_dir = scene_dirs[i];
            bool is_illumination = PatchLoader::isIlluminationScene(scene_dir);

            Accumulator local_overall, local_easy, local_hard, local_tough;
            Accumulator local_illumination_all, local_viewpoint_all;
            Accumulator local_illumination_easy, local_illumination_hard;
            Accumulator local_viewpoint_easy, local_viewpoint_hard;
            int local_total_patches = 0;

            // Evaluate each difficulty level
            std::vector<std::pair<std::string, bool>> difficulties;
            if (config.include_easy) difficulties.emplace_back("easy", true);
            if (config.include_hard) difficulties.emplace_back("hard", true);
            if (config.include_tough) difficulties.emplace_back("tough", true);

            for (const auto& [difficulty, enabled] : difficulties) {
                if (!enabled) continue;

                try {
                    auto result = evaluateScene(scene_dir, *local_extractor, params, config, difficulty);

                    if (result.num_patches > 0) {
                        local_total_patches += result.num_patches;
                        local_overall.add(result);

                        // By difficulty
                        if (difficulty == "easy") {
                            local_easy.add(result);
                            if (is_illumination) local_illumination_easy.add(result);
                            else local_viewpoint_easy.add(result);
                        } else if (difficulty == "hard") {
                            local_hard.add(result);
                            if (is_illumination) local_illumination_hard.add(result);
                            else local_viewpoint_hard.add(result);
                        } else {
                            local_tough.add(result);
                        }

                        // By scene type
                        if (is_illumination) local_illumination_all.add(result);
                        else local_viewpoint_all.add(result);
                    }
                } catch (const std::exception& e) {
                    if (config.verbose) {
                        #pragma omp critical(patch_benchmark_warnings)
                        {
                            std::cerr << "Warning: Failed to evaluate " << scene_dir
                                      << " (" << difficulty << "): " << e.what() << std::endl;
                        }
                    }
                }
            }

            if (local_total_patches > 0) {
                #pragma omp critical(patch_benchmark_accumulate)
                {
                    total_patches += local_total_patches;
                    overall.sum_map += local_overall.sum_map;
                    overall.sum_accuracy += local_overall.sum_accuracy;
                    overall.count += local_overall.count;

                    easy_all.sum_map += local_easy.sum_map;
                    easy_all.sum_accuracy += local_easy.sum_accuracy;
                    easy_all.count += local_easy.count;

                    hard_all.sum_map += local_hard.sum_map;
                    hard_all.sum_accuracy += local_hard.sum_accuracy;
                    hard_all.count += local_hard.count;

                    tough_all.sum_map += local_tough.sum_map;
                    tough_all.sum_accuracy += local_tough.sum_accuracy;
                    tough_all.count += local_tough.count;

                    illumination_all.sum_map += local_illumination_all.sum_map;
                    illumination_all.sum_accuracy += local_illumination_all.sum_accuracy;
                    illumination_all.count += local_illumination_all.count;

                    viewpoint_all.sum_map += local_viewpoint_all.sum_map;
                    viewpoint_all.sum_accuracy += local_viewpoint_all.sum_accuracy;
                    viewpoint_all.count += local_viewpoint_all.count;

                    illumination_easy.sum_map += local_illumination_easy.sum_map;
                    illumination_easy.sum_accuracy += local_illumination_easy.sum_accuracy;
                    illumination_easy.count += local_illumination_easy.count;

                    illumination_hard.sum_map += local_illumination_hard.sum_map;
                    illumination_hard.sum_accuracy += local_illumination_hard.sum_accuracy;
                    illumination_hard.count += local_illumination_hard.count;

                    viewpoint_easy.sum_map += local_viewpoint_easy.sum_map;
                    viewpoint_easy.sum_accuracy += local_viewpoint_easy.sum_accuracy;
                    viewpoint_easy.count += local_viewpoint_easy.count;

                    viewpoint_hard.sum_map += local_viewpoint_hard.sum_map;
                    viewpoint_hard.sum_accuracy += local_viewpoint_hard.sum_accuracy;
                    viewpoint_hard.count += local_viewpoint_hard.count;
                }
            }

            const int current = ++completed;
            if (progress_callback) {
                #pragma omp critical(patch_benchmark_progress)
                {
                    progress_callback(current, static_cast<int>(scene_dirs.size()), scene_dir);
                }
            }
            }
        }

        scenes_processed = completed.load();
    } else {
        // Process each scene sequentially
        for (size_t i = 0; i < scene_dirs.size(); ++i) {
            const auto& scene_dir = scene_dirs[i];
            bool is_illumination = PatchLoader::isIlluminationScene(scene_dir);

            if (progress_callback) {
                progress_callback(static_cast<int>(i + 1), static_cast<int>(scene_dirs.size()), scene_dir);
            }

            // Evaluate each difficulty level
            std::vector<std::pair<std::string, bool>> difficulties;
            if (config.include_easy) difficulties.emplace_back("easy", true);
            if (config.include_hard) difficulties.emplace_back("hard", true);
            if (config.include_tough) difficulties.emplace_back("tough", true);

            for (const auto& [difficulty, enabled] : difficulties) {
                if (!enabled) continue;

                try {
            auto result = evaluateScene(scene_dir, extractor, params, config, difficulty);

                    if (result.num_patches > 0) {
                        total_patches += result.num_patches;
                        overall.add(result);

                        // By difficulty
                        if (difficulty == "easy") {
                            easy_all.add(result);
                            if (is_illumination) illumination_easy.add(result);
                            else viewpoint_easy.add(result);
                        } else if (difficulty == "hard") {
                            hard_all.add(result);
                            if (is_illumination) illumination_hard.add(result);
                            else viewpoint_hard.add(result);
                        } else {
                            tough_all.add(result);
                        }

                        // By scene type
                        if (is_illumination) illumination_all.add(result);
                        else viewpoint_all.add(result);
                    }
                } catch (const std::exception& e) {
                    if (config.verbose) {
                        std::cerr << "Warning: Failed to evaluate " << scene_dir
                                  << " (" << difficulty << "): " << e.what() << std::endl;
                    }
                }
            }

            scenes_processed++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Fill in results
    results.mAP_overall = overall.meanMAP();
    results.accuracy_overall = overall.meanAccuracy();

    results.mAP_easy = easy_all.meanMAP();
    results.mAP_hard = hard_all.meanMAP();
    results.mAP_tough = tough_all.meanMAP();

    results.mAP_illumination = illumination_all.meanMAP();
    results.mAP_viewpoint = viewpoint_all.meanMAP();

    results.mAP_illumination_easy = illumination_easy.meanMAP();
    results.mAP_illumination_hard = illumination_hard.meanMAP();
    results.mAP_viewpoint_easy = viewpoint_easy.meanMAP();
    results.mAP_viewpoint_hard = viewpoint_hard.meanMAP();

    results.num_patches = total_patches;
    results.processing_time_ms = static_cast<double>(duration.count());

    if (config.print_results) {
        printResults(results);
    }

    return results;
}

PatchMetrics::MatchResult HPatchesBenchmark::evaluateScene(
    const std::string& scene_dir,
    IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    const Config& config,
    const std::string& difficulty) {

    // Load the scene
    auto scene = PatchLoader::loadScene(scene_dir, config.color);

    // Get the appropriate patch sets based on difficulty
    const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
    if (difficulty == "easy") {
        target_sets = &scene.easy;
    } else if (difficulty == "hard") {
        target_sets = &scene.hard;
    } else if (difficulty == "tough") {
        target_sets = &scene.tough;
    } else {
        throw std::invalid_argument("Unknown difficulty: " + difficulty);
    }

    if (target_sets->empty()) {
        return {};  // No patches for this difficulty
    }

    // Extract descriptors from reference patches
    cv::Mat ref_descriptors = extractor.extractFromPatches(scene.ref.patches, params);

    // Evaluate against each target set (e1-e5, h1-h5, or t1-t5)
    PatchMetrics::MatchResult combined_result;
    combined_result.num_patches = 0;
    double sum_ap = 0.0;
    double sum_accuracy = 0.0;
    int count = 0;

    for (const auto& [key, patch_set] : *target_sets) {
        if (patch_set.patches.size() != scene.ref.patches.size()) {
            continue;  // Skip mismatched sets
        }

        cv::Mat target_descriptors = extractor.extractFromPatches(patch_set.patches, params);

        auto result = PatchMetrics::computeMatching(ref_descriptors, target_descriptors);

        combined_result.num_patches += result.num_patches;
        combined_result.correct_matches += result.correct_matches;
        sum_ap += result.average_precision;
        sum_accuracy += result.match_accuracy;
        count++;
    }

    if (count > 0) {
        combined_result.average_precision = static_cast<float>(sum_ap / count);
        combined_result.match_accuracy = static_cast<float>(sum_accuracy / count);
    }

    return combined_result;
}

void HPatchesBenchmark::printResults(const Results& results) {
    std::cout << formatResults(results);
}

std::string HPatchesBenchmark::formatResults(const Results& results) {
    std::ostringstream oss;

    oss << "\n";
    oss << "========================================\n";
    oss << "HPatches Patch Benchmark Results\n";
    oss << "========================================\n";
    oss << "Descriptor: " << results.descriptor_name
        << " (" << results.descriptor_dimension << "D)\n";
    oss << "Scenes: " << results.num_scenes
        << ", Patches: " << results.num_patches << "\n";
    oss << "Time: " << std::fixed << std::setprecision(1)
        << results.processing_time_ms / 1000.0 << "s\n";
    oss << "----------------------------------------\n";

    auto pct = [](float v) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << (v * 100.0f) << "%";
        return s.str();
    };

    oss << "Overall mAP:       " << pct(results.mAP_overall) << "\n";
    oss << "Overall Accuracy:  " << pct(results.accuracy_overall) << "\n";
    oss << "----------------------------------------\n";
    oss << "By Difficulty:\n";
    oss << "  Easy:   " << pct(results.mAP_easy) << "\n";
    oss << "  Hard:   " << pct(results.mAP_hard) << "\n";
    if (results.mAP_tough > 0) {
        oss << "  Tough:  " << pct(results.mAP_tough) << "\n";
    }
    oss << "----------------------------------------\n";
    oss << "By Scene Type:\n";
    oss << "  Illumination: " << pct(results.mAP_illumination) << "\n";
    oss << "  Viewpoint:    " << pct(results.mAP_viewpoint) << "\n";
    oss << "----------------------------------------\n";
    oss << "Detailed Breakdown:\n";
    oss << "  Illumination Easy: " << pct(results.mAP_illumination_easy) << "\n";
    oss << "  Illumination Hard: " << pct(results.mAP_illumination_hard) << "\n";
    oss << "  Viewpoint Easy:    " << pct(results.mAP_viewpoint_easy) << "\n";
    oss << "  Viewpoint Hard:    " << pct(results.mAP_viewpoint_hard) << "\n";
    oss << "========================================\n";

    return oss.str();
}

} // namespace thesis_project::benchmark
