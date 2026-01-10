#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace thesis_project {
namespace benchmark {

/**
 * @brief Metrics computation for HPatches patch matching evaluation
 *
 * The HPatches protocol evaluates descriptor performance by:
 * 1. For each reference patch, find its nearest neighbor in the target set
 * 2. A match is correct if the NN index equals the reference index
 * 3. Compute Average Precision based on distance-ranked retrieval
 */
class PatchMetrics {
public:
    /**
     * @brief Result of matching reference patches to target patches
     */
    struct MatchResult {
        int num_patches = 0;           ///< Total number of patches
        int correct_matches = 0;        ///< Number of correct NN matches
        float match_accuracy = 0.0f;    ///< correct_matches / num_patches
        float average_precision = 0.0f; ///< AP based on distance ranking
    };

    /**
     * @brief Compute patch matching metrics
     *
     * For each reference descriptor, find its nearest neighbor in target descriptors.
     * Ground truth: ref[i] should match target[i] (same index = correct).
     *
     * @param ref_descriptors [N, D] matrix of reference descriptors
     * @param target_descriptors [N, D] matrix of target descriptors (same N)
     * @return MatchResult with accuracy and AP
     */
    static MatchResult computeMatching(
        const cv::Mat& ref_descriptors,
        const cv::Mat& target_descriptors);

    /**
     * @brief Compute Average Precision for ranked retrieval
     *
     * @param distances Vector of distances to all targets for each query
     * @param correct_idx Vector of correct target indices (usually just the query index)
     * @return Average Precision score
     */
    static float computeAP(
        const std::vector<std::pair<float, int>>& ranked_results,
        int correct_idx);

    /**
     * @brief Compute mean Average Precision across multiple query/target pairs
     *
     * @param ref_descriptors Reference descriptors
     * @param target_descriptors Target descriptors
     * @return mAP score
     */
    static float computeMAP(
        const cv::Mat& ref_descriptors,
        const cv::Mat& target_descriptors);

    /**
     * @brief Compute False Positive Rate at 95% True Positive Rate (FPR95)
     *
     * Standard HPatches metric: the probability of incorrect match when
     * accepting 95% of correct matches.
     *
     * @param ref_descriptors Reference descriptors
     * @param target_descriptors Target descriptors (positives: same index)
     * @param negative_descriptors Negative descriptors (different patches)
     * @return FPR95 score (lower is better)
     */
    static float computeFPR95(
        const cv::Mat& ref_descriptors,
        const cv::Mat& target_descriptors,
        const cv::Mat& negative_descriptors);

private:
    /**
     * @brief Compute L2 distance between two descriptors
     */
    static float l2Distance(const cv::Mat& desc1, const cv::Mat& desc2);

    /**
     * @brief Find nearest neighbor index and distance
     */
    static std::pair<int, float> findNearestNeighbor(
        const cv::Mat& query,
        const cv::Mat& targets);
};

} // namespace benchmark
} // namespace thesis_project
