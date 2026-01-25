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
namespace database {
class DatabaseManager;
}
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
    struct VerificationTaskPair {
        std::string s1;
        int t1 = 0;
        int idx1 = 0;
        std::string s2;
        int t2 = 0;
        int idx2 = 0;
    };

    struct RetrievalTaskItem {
        std::string s;
        int idx = 0;
    };

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
        bool color = false;                ///< Load color patches (3-channel)
        int num_threads = 4;               ///< Parallel processing threads
        struct TaskConfig {
            std::string mode = "query"; // "query" (per-query AP) or "paper" (Balntas 2017 protocol)
            bool matching = true;
            bool verification = true;
            bool verification_same_seq = true;
            bool verification_diff_seq = true;
            bool retrieval = true;
            int verification_negatives_per_query = 1000;
            int retrieval_negatives_per_query = 1000;
            int verification_num_positives = 200000;
            int verification_num_negatives = 1000000;
            int retrieval_num_queries = 10000;
            int retrieval_num_distractors = 20000;
            unsigned int random_seed = 1337;
            bool preload_descriptors = false;
            std::string preload_scope = "all"; // all or tasks
            bool store_descriptors_to_db = false;
            bool use_cached_descriptors = false;
            std::string descriptor_cache_name;
            int descriptor_cache_id = -1;
            std::string task_source = "random"; // random, db, or csv
            std::string task_set = "hpatches_v1.1";
            std::string task_split = "full";
            std::string tasks_dir;
            std::vector<VerificationTaskPair> verification_pos_pairs;
            std::vector<VerificationTaskPair> verification_neg_inter_pairs;
            std::vector<VerificationTaskPair> verification_neg_intra_pairs;
            std::vector<RetrievalTaskItem> retrieval_queries;
            std::vector<RetrievalTaskItem> retrieval_distractors;
        } tasks;
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

        // Verification (SAMESEQ)
        float verification_same_overall = 0.0f;
        float verification_same_easy = 0.0f;
        float verification_same_hard = 0.0f;
        float verification_same_tough = 0.0f;
        float verification_same_illumination = 0.0f;
        float verification_same_viewpoint = 0.0f;
        float verification_same_illumination_easy = 0.0f;
        float verification_same_illumination_hard = 0.0f;
        float verification_same_viewpoint_easy = 0.0f;
        float verification_same_viewpoint_hard = 0.0f;

        // Verification (DIFFSEQ)
        float verification_diff_overall = 0.0f;
        float verification_diff_easy = 0.0f;
        float verification_diff_hard = 0.0f;
        float verification_diff_tough = 0.0f;
        float verification_diff_illumination = 0.0f;
        float verification_diff_viewpoint = 0.0f;
        float verification_diff_illumination_easy = 0.0f;
        float verification_diff_illumination_hard = 0.0f;
        float verification_diff_viewpoint_easy = 0.0f;
        float verification_diff_viewpoint_hard = 0.0f;

        // Retrieval (DIFFSEQ)
        float retrieval_overall = 0.0f;
        float retrieval_easy = 0.0f;
        float retrieval_hard = 0.0f;
        float retrieval_tough = 0.0f;
        float retrieval_illumination = 0.0f;
        float retrieval_viewpoint = 0.0f;
        float retrieval_illumination_easy = 0.0f;
        float retrieval_illumination_hard = 0.0f;
        float retrieval_viewpoint_easy = 0.0f;
        float retrieval_viewpoint_hard = 0.0f;

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
        database::DatabaseManager* database_manager = nullptr,
        const ProgressCallback& progress_callback = nullptr);

    /**
     * @brief Run the benchmark on a single scene
     * @param scene_dir Path to the scene directory
     * @param extractor Descriptor extractor
     * @param params Descriptor parameters
     * @param difficulty "easy", "hard", or "tough"
     * @param illumination_scenes Pool of illumination scenes for DIFFSEQ sampling
     * @param viewpoint_scenes Pool of viewpoint scenes for DIFFSEQ sampling
     * @return Match result for this scene/difficulty
     */
    struct SceneResults {
        PatchMetrics::MatchResult matching;
        PatchMetrics::MatchResult verification_same;
        PatchMetrics::MatchResult verification_diff;
        PatchMetrics::MatchResult retrieval;
    };

    static SceneResults evaluateScene(
        const std::string& scene_dir,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params,
        const Config& config,
        const std::string& difficulty,
        const std::vector<std::string>& illumination_scenes,
        const std::vector<std::string>& viewpoint_scenes,
        database::DatabaseManager* database_manager = nullptr);

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
