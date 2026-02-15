#include "APMetrics.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

namespace thesis_project::benchmark::metrics {

float computeAPFromLabels(
    const std::vector<std::pair<float, int>>& ranked,
    int positives) {

    if (positives <= 0 || ranked.empty()) {
        return 0.0f;
    }

    int hits = 0;
    double sum_prec = 0.0;

    for (size_t i = 0; i < ranked.size(); ++i) {
        if (ranked[i].second == 1) {
            hits++;
            sum_prec += static_cast<double>(hits) / static_cast<double>(i + 1);
        }
    }

    return hits > 0 ? static_cast<float>(sum_prec / positives) : 0.0f;
}

float computeAPWithIgnore(
    const std::vector<std::pair<float, int>>& ranked,
    int positives) {

    if (positives <= 0 || ranked.empty()) {
        return 0.0f;
    }

    int hits = 0;
    int retrieved = 0;
    double sum_prec = 0.0;

    for (const auto& [score, label] : ranked) {
        // label = 0: ignore this item
        if (label == 0) {
            continue;
        }

        retrieved++;

        // label > 0: positive match
        if (label > 0) {
            hits++;
            sum_prec += static_cast<double>(hits) / static_cast<double>(retrieved);
            if (hits >= positives) {
                break;
            }
        }
    }

    return hits > 0 ? static_cast<float>(sum_prec / positives) : 0.0f;
}

float computeAPTrapz(
    const std::vector<float>& scores,
    const std::vector<int>& labels,
    int numpos) {

    if (scores.empty() || labels.empty() || scores.size() != labels.size()) {
        return 0.0f;
    }

    // Count actual positives
    int pos_count = 0;
    for (int label : labels) {
        if (label == 1) {
            pos_count++;
        }
    }

    if (pos_count == 0) {
        return 0.0f;
    }

    // Work with copies since we may need to add phantom positives
    std::vector<float> work_scores = scores;
    std::vector<int> work_labels = labels;

    // If numpos > actual positives, add missing positives with -inf score
    // (they are considered as not retrieved)
    if (numpos > pos_count) {
        const int extra = numpos - pos_count;
        work_scores.reserve(work_scores.size() + static_cast<size_t>(extra));
        work_labels.reserve(work_labels.size() + static_cast<size_t>(extra));
        for (int i = 0; i < extra; ++i) {
            work_scores.push_back(-std::numeric_limits<float>::infinity());
            work_labels.push_back(1);
        }
        pos_count = numpos;
    }

    // Create sorted permutation by score descending
    std::vector<int> perm(work_scores.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](int a, int b) { return work_scores[a] > work_scores[b]; });

    // Find last valid (non -inf) item
    int last_valid = -1;
    for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
        if (work_scores[perm[i]] > -std::numeric_limits<float>::infinity()) {
            last_valid = i;
        }
    }

    if (last_valid < 0) {
        return 0.0f;
    }

    // Build TP and FP curves
    std::vector<float> tp;
    std::vector<float> fp;
    tp.reserve(static_cast<size_t>(last_valid + 2));
    fp.reserve(static_cast<size_t>(last_valid + 2));
    tp.push_back(0.0f);
    fp.push_back(0.0f);

    int tp_count = 0;
    int fp_count = 0;

    for (int i = 0; i <= last_valid; ++i) {
        const int label = work_labels[perm[i]];
        if (label == 1) {
            tp_count++;
        } else {
            fp_count++;
        }
        tp.push_back(static_cast<float>(tp_count));
        fp.push_back(static_cast<float>(fp_count));
    }

    // Compute recall and precision
    std::vector<float> recall(tp.size());
    std::vector<float> precision(tp.size());
    const float denom_pos = std::max(1.0f, static_cast<float>(pos_count));

    for (size_t i = 0; i < tp.size(); ++i) {
        recall[i] = tp[i] / denom_pos;
        const float denom = std::max(1e-10f, tp[i] + fp[i]);
        precision[i] = tp[i] / denom;
    }

    // Trapezoidal integration
    float ap = 0.0f;
    for (size_t i = 1; i < recall.size(); ++i) {
        const float dr = recall[i] - recall[i - 1];
        ap += dr * (precision[i] + precision[i - 1]) * 0.5f;
    }

    return ap;
}

float l2Distance(const cv::Mat& a, const cv::Mat& b) {
    return static_cast<float>(cv::norm(a, b, cv::NORM_L2));
}

std::vector<float> computeRowDistancesSquared(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.rows != b.rows || a.cols != b.cols) {
        return {};
    }

    // Compute ||a[i] - b[i]||^2 = ||a[i]||^2 + ||b[i]||^2 - 2 * a[i] . b[i]
    cv::Mat a_sq, b_sq, dot;
    cv::multiply(a, a, a_sq);
    cv::multiply(b, b, b_sq);
    cv::reduce(a_sq, a_sq, 1, cv::REDUCE_SUM, CV_32F);
    cv::reduce(b_sq, b_sq, 1, cv::REDUCE_SUM, CV_32F);
    cv::multiply(a, b, dot);
    cv::reduce(dot, dot, 1, cv::REDUCE_SUM, CV_32F);

    std::vector<float> distances(static_cast<size_t>(a.rows));
    for (int i = 0; i < a.rows; ++i) {
        distances[static_cast<size_t>(i)] =
            a_sq.at<float>(i, 0) + b_sq.at<float>(i, 0) - 2.0f * dot.at<float>(i, 0);
    }

    return distances;
}

std::vector<int> sampleUniqueIndices(int total, int exclude, int count, std::mt19937& rng) {
    std::vector<int> indices;

    if (total <= 1 || count <= 0) {
        return indices;
    }

    const int max_count = std::min(count, total - 1);
    indices.reserve(max_count);

    // Build pool excluding the specified index
    std::vector<int> pool;
    pool.reserve(total - 1);
    for (int i = 0; i < total; ++i) {
        if (i != exclude) {
            pool.push_back(i);
        }
    }

    std::shuffle(pool.begin(), pool.end(), rng);
    indices.insert(indices.end(), pool.begin(), pool.begin() + max_count);

    return indices;
}

std::vector<int> sampleIndices(int total, int count, std::mt19937& rng) {
    std::vector<int> indices;

    if (total <= 0 || count <= 0) {
        return indices;
    }

    const int max_count = std::min(count, total);
    indices.reserve(max_count);

    std::vector<int> pool(total);
    std::iota(pool.begin(), pool.end(), 0);
    std::shuffle(pool.begin(), pool.end(), rng);
    indices.insert(indices.end(), pool.begin(), pool.begin() + max_count);

    return indices;
}

} // namespace thesis_project::benchmark::metrics
