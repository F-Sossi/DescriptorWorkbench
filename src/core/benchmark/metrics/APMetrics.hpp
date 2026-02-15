#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <utility>
#include <random>

namespace thesis_project::benchmark::metrics {

/**
 * @brief Compute Average Precision from ranked results with binary labels
 *
 * Standard AP computation: sum of precision@k for each relevant item,
 * divided by total number of positives.
 *
 * @param ranked Vector of (score, label) pairs, sorted by score descending
 *               Labels: 1 = positive, 0 = negative
 * @param positives Total number of positive items (for normalization)
 * @return Average Precision score [0, 1]
 */
float computeAPFromLabels(
    const std::vector<std::pair<float, int>>& ranked,
    int positives);

/**
 * @brief Compute Average Precision with ignore labels (for retrieval)
 *
 * Like computeAPFromLabels but ignores items with label=0.
 * Used in HPatches retrieval where:
 * - label = 1: positive match
 * - label = -1: negative (distractor)
 * - label = 0: ignore (excluded from ranking)
 *
 * @param ranked Vector of (score, label) pairs, sorted by score descending
 * @param positives Number of positive items
 * @return Average Precision score [0, 1]
 */
float computeAPWithIgnore(
    const std::vector<std::pair<float, int>>& ranked,
    int positives);

/**
 * @brief Compute AP using trapezoidal integration of PR curve
 *
 * Used for HPatches imbalanced protocol where positives may be added
 * with -infinity scores to account for missed detections.
 *
 * @param scores Vector of scores (higher = more likely positive)
 * @param labels Vector of labels (1 = positive, 0 = negative)
 * @param numpos Expected number of positives (if > actual, missing positives
 *               are added with -inf score). Use -1 to use actual count.
 * @return Average Precision score [0, 1]
 */
float computeAPTrapz(
    const std::vector<float>& scores,
    const std::vector<int>& labels,
    int numpos = -1);

/**
 * @brief Compute L2 distance between two descriptor rows
 *
 * @param a First descriptor (1xD or Dx1 matrix)
 * @param b Second descriptor (same size as a)
 * @return Euclidean distance
 */
float l2Distance(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Compute squared L2 distances for corresponding rows
 *
 * Efficiently computes ||a[i] - b[i]||^2 for all rows.
 *
 * @param a First descriptor matrix (NxD)
 * @param b Second descriptor matrix (NxD)
 * @return Vector of N squared distances
 */
std::vector<float> computeRowDistancesSquared(const cv::Mat& a, const cv::Mat& b);

/**
 * @brief Sample unique indices excluding one value
 *
 * @param total Total number of indices to sample from [0, total)
 * @param exclude Index to exclude from sampling
 * @param count Number of indices to sample
 * @param rng Random number generator
 * @return Vector of sampled indices
 */
std::vector<int> sampleUniqueIndices(int total, int exclude, int count, std::mt19937& rng);

/**
 * @brief Sample indices without replacement
 *
 * @param total Total number of indices [0, total)
 * @param count Number to sample
 * @param rng Random number generator
 * @return Vector of sampled indices
 */
std::vector<int> sampleIndices(int total, int count, std::mt19937& rng);

} // namespace thesis_project::benchmark::metrics
