#include "MatchingTask.hpp"
#include "../metrics/APMetrics.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <filesystem>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace thesis_project::benchmark::tasks {

namespace {

/// Accumulator for computing mean metrics (supports merging for parallel reduction)
struct Accumulator {
    double sum_mAP = 0.0;
    double sum_accuracy = 0.0;
    int count = 0;
    int total_patches = 0;

    void add(float mAP, float accuracy, int patches) {
        sum_mAP += mAP;
        sum_accuracy += accuracy;
        count++;
        total_patches += patches;
    }

    void merge(const Accumulator& other) {
        sum_mAP += other.sum_mAP;
        sum_accuracy += other.sum_accuracy;
        count += other.count;
        total_patches += other.total_patches;
    }

    float meanMAP() const {
        return count > 0 ? static_cast<float>(sum_mAP / count) : 0.0f;
    }

    float meanAccuracy() const {
        return count > 0 ? static_cast<float>(sum_accuracy / count) : 0.0f;
    }
};

} // anonymous namespace

MatchingResults MatchingTask::run(
    const DescriptorBank& bank,
    const Config& config,
    const std::vector<std::string>& scene_dirs,
    const ProgressCallback& progress) {

    MatchingResults results;
    const int total = static_cast<int>(scene_dirs.size());

    // Final accumulators (will be merged from thread-local ones)
    Accumulator overall;
    Accumulator easy_acc, hard_acc, tough_acc;
    Accumulator illum_acc, view_acc;
    Accumulator illum_easy_acc, illum_hard_acc;
    Accumulator view_easy_acc, view_hard_acc;

#ifdef _OPENMP
    #pragma omp parallel
    {
        // Thread-local accumulators
        Accumulator t_overall;
        Accumulator t_easy, t_hard, t_tough;
        Accumulator t_illum, t_view;
        Accumulator t_illum_easy, t_illum_hard;
        Accumulator t_view_easy, t_view_hard;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < total; ++i) {
            const auto& scene_dir = scene_dirs[i];
            std::string scene_name = std::filesystem::path(scene_dir).filename().string();
            bool is_illumination = isIlluminationScene(scene_name);

            if (progress) {
                #pragma omp critical
                progress(i + 1, total, scene_name);
            }

            // Process each enabled difficulty
            std::vector<std::pair<std::string, bool>> difficulties = {
                {"easy", config.include_easy},
                {"hard", config.include_hard},
                {"tough", config.include_tough}
            };

            for (const auto& [difficulty, enabled] : difficulties) {
                if (!enabled) continue;

                const cv::Mat& ref_desc = bank.get(scene_name, difficulty, "ref");
                if (ref_desc.empty()) continue;

                std::vector<std::string> targets;
                std::string prefix = PatchKey::difficultyPrefix(difficulty);
                for (int j = 1; j <= 5; ++j) {
                    targets.push_back(prefix + std::to_string(j));
                }

                for (const auto& target : targets) {
                    const cv::Mat& target_desc = bank.get(scene_name, difficulty, target);
                    if (target_desc.empty() || target_desc.rows != ref_desc.rows) continue;

                    float accuracy = 0.0f;
                    float mAP = computeSceneMAP(ref_desc, target_desc, config.matching, &accuracy);
                    int patches = ref_desc.rows;

                    t_overall.add(mAP, accuracy, patches);

                    if (difficulty == "easy") {
                        t_easy.add(mAP, accuracy, patches);
                        if (is_illumination) t_illum_easy.add(mAP, accuracy, patches);
                        else t_view_easy.add(mAP, accuracy, patches);
                    } else if (difficulty == "hard") {
                        t_hard.add(mAP, accuracy, patches);
                        if (is_illumination) t_illum_hard.add(mAP, accuracy, patches);
                        else t_view_hard.add(mAP, accuracy, patches);
                    } else {
                        t_tough.add(mAP, accuracy, patches);
                    }

                    if (is_illumination) {
                        t_illum.add(mAP, accuracy, patches);
                    } else {
                        t_view.add(mAP, accuracy, patches);
                    }
                }
            }
        }

        // Merge thread-local accumulators into global ones
        #pragma omp critical
        {
            overall.merge(t_overall);
            easy_acc.merge(t_easy);
            hard_acc.merge(t_hard);
            tough_acc.merge(t_tough);
            illum_acc.merge(t_illum);
            view_acc.merge(t_view);
            illum_easy_acc.merge(t_illum_easy);
            illum_hard_acc.merge(t_illum_hard);
            view_easy_acc.merge(t_view_easy);
            view_hard_acc.merge(t_view_hard);
        }
    }
#else
    // Non-OpenMP fallback
    for (int i = 0; i < total; ++i) {
        const auto& scene_dir = scene_dirs[i];
        std::string scene_name = std::filesystem::path(scene_dir).filename().string();
        bool is_illumination = isIlluminationScene(scene_name);

        if (progress) {
            progress(i + 1, total, scene_name);
        }

        std::vector<std::pair<std::string, bool>> difficulties = {
            {"easy", config.include_easy},
            {"hard", config.include_hard},
            {"tough", config.include_tough}
        };

        for (const auto& [difficulty, enabled] : difficulties) {
            if (!enabled) continue;

            const cv::Mat& ref_desc = bank.get(scene_name, difficulty, "ref");
            if (ref_desc.empty()) continue;

            std::vector<std::string> targets;
            std::string prefix = PatchKey::difficultyPrefix(difficulty);
            for (int j = 1; j <= 5; ++j) {
                targets.push_back(prefix + std::to_string(j));
            }

            for (const auto& target : targets) {
                const cv::Mat& target_desc = bank.get(scene_name, difficulty, target);
                if (target_desc.empty() || target_desc.rows != ref_desc.rows) continue;

                float accuracy = 0.0f;
                float mAP = computeSceneMAP(ref_desc, target_desc, config.matching, &accuracy);
                int patches = ref_desc.rows;

                overall.add(mAP, accuracy, patches);

                if (difficulty == "easy") {
                    easy_acc.add(mAP, accuracy, patches);
                    if (is_illumination) illum_easy_acc.add(mAP, accuracy, patches);
                    else view_easy_acc.add(mAP, accuracy, patches);
                } else if (difficulty == "hard") {
                    hard_acc.add(mAP, accuracy, patches);
                    if (is_illumination) illum_hard_acc.add(mAP, accuracy, patches);
                    else view_hard_acc.add(mAP, accuracy, patches);
                } else {
                    tough_acc.add(mAP, accuracy, patches);
                }

                if (is_illumination) {
                    illum_acc.add(mAP, accuracy, patches);
                } else {
                    view_acc.add(mAP, accuracy, patches);
                }
            }
        }
    }
#endif

    // Populate results
    results.mAP_overall = overall.meanMAP();
    results.accuracy_overall = overall.meanAccuracy();
    results.num_patches = overall.total_patches;

    results.mAP_easy = easy_acc.meanMAP();
    results.mAP_hard = hard_acc.meanMAP();
    results.mAP_tough = tough_acc.meanMAP();

    results.accuracy_easy = easy_acc.meanAccuracy();
    results.accuracy_hard = hard_acc.meanAccuracy();
    results.accuracy_tough = tough_acc.meanAccuracy();

    results.mAP_illumination = illum_acc.meanMAP();
    results.mAP_viewpoint = view_acc.meanMAP();

    results.mAP_illumination_easy = illum_easy_acc.meanMAP();
    results.mAP_illumination_hard = illum_hard_acc.meanMAP();
    results.mAP_viewpoint_easy = view_easy_acc.meanMAP();
    results.mAP_viewpoint_hard = view_hard_acc.meanMAP();

    return results;
}

float MatchingTask::computeSceneMAP(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const MatchingConfig& matching_config,
    float* accuracy_out) {

    if (ref_desc.empty() || target_desc.empty()) {
        if (accuracy_out) *accuracy_out = 0.0f;
        return 0.0f;
    }

    if (ref_desc.rows != target_desc.rows) {
        if (accuracy_out) *accuracy_out = 0.0f;
        return 0.0f;
    }

    const int N = ref_desc.rows;

    // HPatches matching protocol (global per-scene AP):
    // 1. For each query i, find nearest neighbor in target
    // 2. Check if NN index == query index (ground truth)
    // 3. Compute single AP over all N queries with numpos=N
    //    (expecting all N queries to find their correct match)

    std::vector<float> nn_scores;  // -distance to NN for each query
    std::vector<int> nn_labels;    // 1 if NN is correct, 0 otherwise
    nn_scores.reserve(N);
    nn_labels.reserve(N);

    int correct = 0;

    for (int q = 0; q < N; ++q) {
        cv::Mat query = ref_desc.row(q);

        // Find nearest neighbor
        float min_dist = std::numeric_limits<float>::max();
        int nn_idx = -1;

        for (int t = 0; t < target_desc.rows; ++t) {
            float dist = metrics::l2Distance(query, target_desc.row(t));
            if (dist < min_dist) {
                min_dist = dist;
                nn_idx = t;
            }
        }

        // Check if NN is correct (ground truth: query i should match target i)
        bool is_correct = (nn_idx == q);
        if (is_correct) {
            correct++;
        }

        // Store score and label for global AP computation
        // Score = -distance (higher = more similar)
        nn_scores.push_back(-min_dist);
        nn_labels.push_back(is_correct ? 1 : 0);
    }

    if (accuracy_out) {
        *accuracy_out = static_cast<float>(correct) / static_cast<float>(N);
    }

    // Compute global AP with numpos=N (expect all queries to find correct match)
    // This matches the HPatches reference: pr,rc,ap = metrics.pr(-m_d, m_l, numpos=m_l.shape[0])
    return metrics::computeAPTrapz(nn_scores, nn_labels, N);
}

bool MatchingTask::isIlluminationScene(const std::string& scene) {
    return scene.size() > 2 && scene[0] == 'i' && scene[1] == '_';
}

} // namespace thesis_project::benchmark::tasks
