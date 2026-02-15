#pragma once

#include "../BenchmarkTypes.hpp"
#include "../DescriptorBank.hpp"
#include <vector>
#include <string>

namespace thesis_project::benchmark::tasks {

/**
 * @brief Results from verification task evaluation
 */
struct VerificationResults {
    // Same-sequence results (negatives from same sequence, different patch)
    float same_seq_overall = 0.0f;
    float same_seq_easy = 0.0f;
    float same_seq_hard = 0.0f;
    float same_seq_tough = 0.0f;
    float same_seq_illumination = 0.0f;
    float same_seq_viewpoint = 0.0f;

    // Different-sequence results (negatives from different sequences)
    float diff_seq_overall = 0.0f;
    float diff_seq_easy = 0.0f;
    float diff_seq_hard = 0.0f;
    float diff_seq_tough = 0.0f;
    float diff_seq_illumination = 0.0f;
    float diff_seq_viewpoint = 0.0f;
};

/**
 * @brief Verification task for HPatches patch benchmark
 *
 * The verification task evaluates pair classification:
 * - Given pairs of patches, determine if they are the same keypoint
 * - Positive: same patch index across different images (same 3D point)
 * - Negative same-seq: different patches from same sequence
 * - Negative diff-seq: patches from different sequences
 *
 * Uses the HPatches paper protocol with pre-defined task pairs.
 */
class VerificationTask {
public:
    /**
     * @brief Run verification task evaluation
     *
     * Uses task pairs from config (loaded from CSV files) to evaluate.
     *
     * @param bank DescriptorBank containing required descriptors
     * @param config Benchmark configuration with task pairs
     * @return VerificationResults with AP breakdown
     */
    static VerificationResults run(
        const DescriptorBank& bank,
        const Config& config);

private:
    /**
     * @brief Run verification using task file pairs
     *
     * @param bank Descriptor bank
     * @param pos_pairs Positive pairs (same patch across images)
     * @param neg_pairs Negative pairs (different patches)
     * @param difficulty "easy", "hard", or "tough"
     * @param split Filter: "full", "illum", or "view"
     * @return AP score for this evaluation
     */
    static float runFromTasks(
        const DescriptorBank& bank,
        const std::vector<VerificationTaskPair>& pos_pairs,
        const std::vector<VerificationTaskPair>& neg_pairs,
        const std::string& difficulty,
        const std::string& split);

    /**
     * @brief Check if scene matches split filter
     */
    static bool matchesSplit(const std::string& scene, const std::string& split);
};

} // namespace thesis_project::benchmark::tasks
