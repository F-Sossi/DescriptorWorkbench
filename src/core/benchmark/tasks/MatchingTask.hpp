#pragma once

#include "../BenchmarkTypes.hpp"
#include "../DescriptorBank.hpp"
#include "../PatchScope.hpp"
#include <vector>
#include <string>

namespace thesis_project::benchmark::tasks {

/**
 * @brief Results from matching task evaluation
 */
struct MatchingResults {
    // Overall metrics
    float mAP_overall = 0.0f;
    float accuracy_overall = 0.0f;
    int num_patches = 0;

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

    float accuracy_easy = 0.0f;
    float accuracy_hard = 0.0f;
    float accuracy_tough = 0.0f;
};

/**
 * @brief Matching task for HPatches patch benchmark
 *
 * The matching task evaluates descriptor discriminability:
 * - For each reference patch, find its nearest neighbor in the target set
 * - Ground truth: ref[i] should match target[i] (same index)
 * - Compute success rate and mAP
 *
 * This follows the HPatches paper protocol for patch matching evaluation.
 */
class MatchingTask {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& scene)>;

    /**
     * @brief Run matching task evaluation
     *
     * Evaluates matching across all scenes and difficulties in the bank.
     * For each scene/difficulty, computes mAP for ref vs e1-e5/h1-h5/t1-t5.
     *
     * @param bank DescriptorBank containing all required descriptors
     * @param config Benchmark configuration
     * @param scene_dirs List of scene directories (for categorization)
     * @param progress Optional progress callback
     * @return MatchingResults with mAP breakdown by difficulty and scene type
     */
    static MatchingResults run(
        const DescriptorBank& bank,
        const Config& config,
        const std::vector<std::string>& scene_dirs,
        const ProgressCallback& progress = nullptr);

private:
    /**
     * @brief Compute mAP and accuracy for a single ref/target pair
     *
     * @param ref_desc Reference descriptors [N x D]
     * @param target_desc Target descriptors [N x D]
     * @param accuracy_out Output: matching accuracy (correct NN / total)
     * @return Average Precision score
     */
    static float computeSceneMAP(
        const cv::Mat& ref_desc,
        const cv::Mat& target_desc,
        float* accuracy_out = nullptr);

    /**
     * @brief Check if a scene is an illumination scene (i_*)
     */
    static bool isIlluminationScene(const std::string& scene);
};

} // namespace thesis_project::benchmark::tasks
