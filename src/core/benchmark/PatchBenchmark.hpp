#pragma once

#include "BenchmarkTypes.hpp"
#include "PatchScope.hpp"
#include "DescriptorBank.hpp"
#include "ResultsFormatter.hpp"
#include "tasks/MatchingTask.hpp"
#include "tasks/VerificationTask.hpp"
#include "tasks/RetrievalTask.hpp"
#include "core/patches/PatchDescriptorExtractor.hpp"
#include <thesis_project/types.hpp>
#include <functional>

namespace thesis_project {
namespace database { class DatabaseManager; }
namespace benchmark {

/**
 * @brief Main orchestrator for HPatches patch benchmark
 *
 * PatchBenchmark provides a clean 4-phase pipeline:
 * 1. Initialize: Build scope, load task files
 * 2. Extract: Load/extract all required descriptors (once)
 * 3. Execute: Run matching, verification, retrieval tasks
 * 4. Output: Format and optionally store results
 *
 * Usage:
 * @code
 * Config config;
 * config.patches_dir = "../hpatches-release";
 * config.matching_enabled = true;
 * config.verification_enabled = true;
 * config.retrieval_enabled = true;
 * // ... load task pairs into config ...
 *
 * auto extractor = createDescriptorExtractor("sift");
 * DescriptorParams params;
 *
 * auto results = PatchBenchmark::run(config, *extractor, params, db);
 * @endcode
 */
class PatchBenchmark {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& message)>;

    /**
     * @brief Run the complete benchmark pipeline
     *
     * This is the main entry point. It:
     * 1. Builds PatchScope based on enabled tasks
     * 2. Creates DescriptorBank and loads/extracts descriptors
     * 3. Runs enabled tasks (matching, verification, retrieval)
     * 4. Returns aggregated results
     *
     * @param config Benchmark configuration (tasks, paths, options)
     * @param extractor Descriptor extractor to use
     * @param params Descriptor extraction parameters
     * @param db Optional database manager for caching/storage
     * @param progress Optional progress callback
     * @return Results struct with all computed metrics
     */
    static Results run(
        const Config& config,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params,
        database::DatabaseManager* db = nullptr,
        const ProgressCallback& progress = nullptr);

    /**
     * @brief Format results as human-readable string
     */
    static std::string formatResults(const Results& results) {
        return ResultsFormatter::format(results);
    }

    /**
     * @brief Print results to stdout
     */
    static void printResults(const Results& results) {
        ResultsFormatter::print(results);
    }

private:
    /**
     * @brief Merge matching results into main results
     */
    static void mergeMatchingResults(Results& results, const tasks::MatchingResults& matching);

    /**
     * @brief Merge verification results into main results
     */
    static void mergeVerificationResults(Results& results, const tasks::VerificationResults& verification);

    /**
     * @brief Merge retrieval results into main results
     */
    static void mergeRetrievalResults(Results& results, const tasks::RetrievalResults& retrieval);
};

} // namespace benchmark
} // namespace thesis_project
