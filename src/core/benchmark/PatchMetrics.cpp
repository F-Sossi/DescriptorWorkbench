#include "PatchMetrics.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>


namespace thesis_project::benchmark {

PatchMetrics::MatchResult PatchMetrics::computeMatching(
    const cv::Mat& ref_descriptors,
    const cv::Mat& target_descriptors) {

    MatchResult result;

    if (ref_descriptors.empty() || target_descriptors.empty()) {
        return result;
    }

    if (ref_descriptors.rows != target_descriptors.rows) {
        throw std::runtime_error("PatchMetrics: descriptor count mismatch");
    }

    result.num_patches = ref_descriptors.rows;
    result.correct_matches = 0;

    // For each reference descriptor, find nearest neighbor in targets
    for (int i = 0; i < ref_descriptors.rows; ++i) {
        cv::Mat query = ref_descriptors.row(i);
        auto [nn_idx, nn_dist] = findNearestNeighbor(query, target_descriptors);

        // Correct if NN index matches query index
        if (nn_idx == i) {
            result.correct_matches++;
        }
    }

    result.match_accuracy = static_cast<float>(result.correct_matches) /
                           static_cast<float>(result.num_patches);

    // Compute mAP
    result.average_precision = computeMAP(ref_descriptors, target_descriptors);

    return result;
}

float PatchMetrics::computeAP(
    const std::vector<std::pair<float, int>>& ranked_results,
    int correct_idx) {

    // ranked_results: [(distance, index), ...] sorted by distance ascending
    // correct_idx: the index that is considered correct for this query

    int num_relevant = 0;
    float sum_precision = 0.0f;

    for (size_t rank = 0; rank < ranked_results.size(); ++rank) {
        if (ranked_results[rank].second == correct_idx) {
            num_relevant++;
            // Precision at this rank
            float precision = static_cast<float>(num_relevant) / static_cast<float>(rank + 1);
            sum_precision += precision;
        }
    }

    // For single-query retrieval, there's exactly 1 relevant item
    // AP = precision at the rank of the correct match
    if (num_relevant == 0) {
        return 0.0f;
    }

    return sum_precision / static_cast<float>(num_relevant);
}

float PatchMetrics::computeMAP(
    const cv::Mat& ref_descriptors,
    const cv::Mat& target_descriptors) {

    if (ref_descriptors.empty() || target_descriptors.empty()) {
        return 0.0f;
    }

    float sum_ap = 0.0f;
    const int num_queries = ref_descriptors.rows;

    for (int q = 0; q < num_queries; ++q) {
        cv::Mat query = ref_descriptors.row(q);

        // Compute distances to all targets
        std::vector<std::pair<float, int>> distances;
        distances.reserve(target_descriptors.rows);

        for (int t = 0; t < target_descriptors.rows; ++t) {
            float dist = l2Distance(query, target_descriptors.row(t));
            distances.emplace_back(dist, t);
        }

        // Sort by distance (ascending)
        std::sort(distances.begin(), distances.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // The correct target has the same index as the query
        float ap = computeAP(distances, q);
        sum_ap += ap;
    }

    return sum_ap / static_cast<float>(num_queries);
}

float PatchMetrics::computeFPR95(
    const cv::Mat& ref_descriptors,
    const cv::Mat& target_descriptors,
    const cv::Mat& negative_descriptors) {

    if (ref_descriptors.empty() || target_descriptors.empty() || negative_descriptors.empty()) {
        return 1.0f;
    }

    // Compute positive distances (ref[i] to target[i])
    std::vector<float> positive_distances;
    positive_distances.reserve(ref_descriptors.rows);

    for (int i = 0; i < ref_descriptors.rows; ++i) {
        float dist = l2Distance(ref_descriptors.row(i), target_descriptors.row(i));
        positive_distances.push_back(dist);
    }

    // Compute negative distances (ref to negatives)
    std::vector<float> negative_distances;
    negative_distances.reserve(ref_descriptors.rows * negative_descriptors.rows);

    for (int i = 0; i < ref_descriptors.rows; ++i) {
        for (int j = 0; j < negative_descriptors.rows; ++j) {
            float dist = l2Distance(ref_descriptors.row(i), negative_descriptors.row(j));
            negative_distances.push_back(dist);
        }
    }

    // Sort positive distances to find 95% threshold
    std::sort(positive_distances.begin(), positive_distances.end());
    int idx_95 = static_cast<int>(0.95f * static_cast<float>(positive_distances.size()));
    float threshold = positive_distances[idx_95];

    // Count negatives below threshold (false positives)
    int false_positives = 0;
    for (float neg_dist : negative_distances) {
        if (neg_dist < threshold) {
            false_positives++;
        }
    }

    return static_cast<float>(false_positives) / static_cast<float>(negative_distances.size());
}

float PatchMetrics::l2Distance(const cv::Mat& desc1, const cv::Mat& desc2) {
    cv::Mat diff = desc1 - desc2;
    return static_cast<float>(cv::norm(diff, cv::NORM_L2));
}

std::pair<int, float> PatchMetrics::findNearestNeighbor(
    const cv::Mat& query,
    const cv::Mat& targets) {

    int best_idx = -1;
    float best_dist = std::numeric_limits<float>::max();

    for (int i = 0; i < targets.rows; ++i) {
        float dist = l2Distance(query, targets.row(i));
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    return {best_idx, best_dist};
}

} // namespace thesis_project::benchmark

