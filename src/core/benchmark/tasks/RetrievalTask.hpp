#pragma once

#include "../BenchmarkTypes.hpp"
#include "../DescriptorBank.hpp"
#include <vector>
#include <string>

namespace thesis_project::benchmark::tasks {

/**
 * @brief Results from retrieval task evaluation
 */
struct RetrievalResults {
    float mAP_overall = 0.0f;
    float mAP_easy = 0.0f;
    float mAP_hard = 0.0f;
    float mAP_tough = 0.0f;
    float mAP_illumination = 0.0f;
    float mAP_viewpoint = 0.0f;
};

/**
 * @brief Retrieval task for HPatches patch benchmark
 *
 * The retrieval task evaluates descriptor ranking quality:
 * - Query: a reference patch
 * - Targets: 5 positive matches (same patch across transformations)
 * - Distractors: many negative patches (from same or different sequences)
 *
 * Uses three-label AP:
 * - label = 1: positive (correct match)
 * - label = -1: negative (distractor)
 * - label = 0: ignore (not counted in ranking)
 */
class RetrievalTask {
public:
    /**
     * @brief Run retrieval task evaluation
     *
     * Uses task files (queries + distractors) from config to evaluate.
     *
     * @param bank DescriptorBank containing required descriptors
     * @param config Benchmark configuration with task data
     * @return RetrievalResults with mAP breakdown
     */
    static RetrievalResults run(
        const DescriptorBank& bank,
        const Config& config);

private:
    /**
     * @brief Run retrieval using task file data
     *
     * @param bank Descriptor bank
     * @param queries Query items (patches to search for)
     * @param distractors Distractor items (negative pool)
     * @param difficulty "easy", "hard", or "tough"
     * @param split Filter: "full", "illum", or "view"
     * @return mAP score for this evaluation
     */
    static float runFromTasks(
        const DescriptorBank& bank,
        const std::vector<RetrievalTaskItem>& queries,
        const std::vector<RetrievalTaskItem>& distractors,
        const std::string& difficulty,
        const std::string& split);

    /**
     * @brief Check if scene matches split filter
     */
    static bool matchesSplit(const std::string& scene, const std::string& split);
};

} // namespace thesis_project::benchmark::tasks
