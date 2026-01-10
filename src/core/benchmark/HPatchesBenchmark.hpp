#pragma once

#include "PatchMetrics.hpp"
#include "core/patches/PatchLoader.hpp"
#include "core/patches/PatchDescriptorExtractor.hpp"
#include <thesis_project/types.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace thesis_project {
namespace benchmark {

/**
 * @brief Orchestrates the HPatches patch benchmark
 *
 * Runs descriptor evaluation on pre-extracted HPatches patches:
 * 1. Loads patches from all scenes
 * 2. Computes descriptors using the provided extractor
 * 3. Evaluates matching performance (mAP, accuracy)
 * 4. Reports results broken down by difficulty and scene type
 */
class HPatchesBenchmark {
public:
    /**
     * @brief Configuration for the benchmark run
     */
    struct Config {
        std::string patches_dir;           ///< Path to hpatches-release
        std::vector<std::string> scenes;   ///< Specific scenes (empty = all)
        bool include_easy = true;          ///< Include easy patches (e1-e5)
        bool include_hard = true;          ///< Include hard patches (h1-h5)
        bool include_tough = true;         ///< Include tough patches (t1-t5)
        bool verbose = true;               ///< Print progress
        bool print_results = true;         ///< Print results summary
        int num_threads = 4;               ///< Parallel processing threads
    };

    /**
     * @brief Results from a benchmark run
     */
    struct Results {
        std::string descriptor_name;
        int descriptor_dimension = 0;

        // Overall metrics
        float mAP_overall = 0.0f;
        float accuracy_overall = 0.0f;

        // By difficulty
        float mAP_easy = 0.0f;
        float mAP_hard = 0.0f;
        float mAP_tough = 0.0f;

        // By scene type
        float mAP_illumination = 0.0f;
        float mAP_viewpoint = 0.0f;

        // Detailed breakdown
        float mAP_illumination_easy = 0.0f;
        float mAP_illumination_hard = 0.0f;
        float mAP_viewpoint_easy = 0.0f;
        float mAP_viewpoint_hard = 0.0f;

        // Statistics
        int num_scenes = 0;
        int num_patches = 0;
        double processing_time_ms = 0.0;
    };

    /**
     * @brief Progress callback type
     */
    using ProgressCallback = std::function<void(int current, int total, const std::string& scene)>;

    /**
     * @brief Run the benchmark
     * @param config Benchmark configuration
     * @param extractor Descriptor extractor to evaluate
     * @param params Descriptor parameters
     * @param progress_callback Optional progress callback
     * @return Benchmark results
     */
    static Results run(
        const Config& config,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params,
        ProgressCallback progress_callback = nullptr);

    /**
     * @brief Run the benchmark on a single scene
     * @param scene_dir Path to the scene directory
     * @param extractor Descriptor extractor
     * @param params Descriptor parameters
     * @param difficulty "easy", "hard", or "tough"
     * @return Match result for this scene/difficulty
     */
    static PatchMetrics::MatchResult evaluateScene(
        const std::string& scene_dir,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params,
        const std::string& difficulty);

    /**
     * @brief Print results to console
     */
    static void printResults(const Results& results);

    /**
     * @brief Format results as a string table
     */
    static std::string formatResults(const Results& results);

private:
    /**
     * @brief Accumulator for collecting results across scenes
     */
    struct Accumulator {
        double sum_map = 0.0;
        double sum_accuracy = 0.0;
        int count = 0;

        void add(const PatchMetrics::MatchResult& result) {
            sum_map += result.average_precision;
            sum_accuracy += result.match_accuracy;
            count++;
        }

        float meanMAP() const {
            return count > 0 ? static_cast<float>(sum_map / count) : 0.0f;
        }

        float meanAccuracy() const {
            return count > 0 ? static_cast<float>(sum_accuracy / count) : 0.0f;
        }
    };
};

} // namespace benchmark
} // namespace thesis_project
