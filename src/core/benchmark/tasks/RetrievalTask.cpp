#include "RetrievalTask.hpp"
#include "../metrics/APMetrics.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <vector>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace thesis_project::benchmark::tasks {

RetrievalResults RetrievalTask::run(
    const DescriptorBank& bank,
    const Config& config) {

    RetrievalResults results;

    if (config.retrieval_queries.empty() || config.retrieval_distractors.empty()) {
        return results;
    }

    // Helper to compute mean of three values
    auto mean3 = [](float a, float b, float c) {
        return (a + b + c) / 3.0f;
    };

    // Compute for each difficulty
    results.mAP_easy = config.include_easy
        ? runFromTasks(bank, config.retrieval_queries,
                      config.retrieval_distractors, "easy", "full")
        : 0.0f;

    results.mAP_hard = config.include_hard
        ? runFromTasks(bank, config.retrieval_queries,
                      config.retrieval_distractors, "hard", "full")
        : 0.0f;

    results.mAP_tough = config.include_tough
        ? runFromTasks(bank, config.retrieval_queries,
                      config.retrieval_distractors, "tough", "full")
        : 0.0f;

    results.mAP_overall = mean3(
        results.mAP_easy,
        results.mAP_hard,
        results.mAP_tough);

    // By scene type
    results.mAP_illumination = mean3(
        config.include_easy ? runFromTasks(bank, config.retrieval_queries,
                                           config.retrieval_distractors, "easy", "illum") : 0.0f,
        config.include_hard ? runFromTasks(bank, config.retrieval_queries,
                                           config.retrieval_distractors, "hard", "illum") : 0.0f,
        config.include_tough ? runFromTasks(bank, config.retrieval_queries,
                                            config.retrieval_distractors, "tough", "illum") : 0.0f);

    results.mAP_viewpoint = mean3(
        config.include_easy ? runFromTasks(bank, config.retrieval_queries,
                                           config.retrieval_distractors, "easy", "view") : 0.0f,
        config.include_hard ? runFromTasks(bank, config.retrieval_queries,
                                           config.retrieval_distractors, "hard", "view") : 0.0f,
        config.include_tough ? runFromTasks(bank, config.retrieval_queries,
                                            config.retrieval_distractors, "tough", "view") : 0.0f);

    return results;
}

float RetrievalTask::runFromTasks(
    const DescriptorBank& bank,
    const std::vector<RetrievalTaskItem>& queries,
    const std::vector<RetrievalTaskItem>& distractors,
    const std::string& difficulty,
    const std::string& split) {

    if (queries.empty() || distractors.empty()) {
        return 0.0f;
    }

    // Build distractor descriptor matrix and index mapping
    std::vector<cv::Mat> distractor_rows;
    std::vector<std::string> distractor_scenes;
    distractor_rows.reserve(distractors.size());
    distractor_scenes.reserve(distractors.size());

    for (const auto& item : distractors) {
        if (!matchesSplit(item.s, split)) continue;

        const cv::Mat& desc = bank.get(item.s, difficulty, "ref");
        if (desc.empty() || item.idx >= desc.rows) continue;

        distractor_rows.push_back(desc.row(item.idx));
        distractor_scenes.push_back(item.s);
    }

    if (distractor_rows.empty()) {
        return 0.0f;
    }

    const int num_queries = static_cast<int>(queries.size());
    double sum_ap = 0.0;
    int query_count = 0;

#ifdef _OPENMP
    #pragma omp parallel reduction(+:sum_ap, query_count)
    {
        #pragma omp for schedule(dynamic)
        for (int q = 0; q < num_queries; ++q) {
            const auto& query_item = queries[q];
            if (!matchesSplit(query_item.s, split)) continue;

            const cv::Mat& query_scene_desc = bank.get(query_item.s, difficulty, "ref");
            if (query_scene_desc.empty() || query_item.idx >= query_scene_desc.rows) continue;

            cv::Mat query_desc = query_scene_desc.row(query_item.idx);

            std::string prefix = PatchKey::difficultyPrefix(difficulty);
            std::vector<cv::Mat> positive_descs;
            for (int i = 1; i <= 5; ++i) {
                std::string target = prefix + std::to_string(i);
                const cv::Mat& target_desc = bank.get(query_item.s, difficulty, target);
                if (!target_desc.empty() && query_item.idx < target_desc.rows) {
                    positive_descs.push_back(target_desc.row(query_item.idx));
                }
            }

            if (positive_descs.empty()) continue;

            const int num_positives = static_cast<int>(positive_descs.size());

            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(positive_descs.size() + distractor_rows.size());

            for (const auto& pos_desc : positive_descs) {
                float dist = metrics::l2Distance(query_desc, pos_desc);
                ranked.emplace_back(-dist, 1);
            }

            for (size_t d = 0; d < distractor_rows.size(); ++d) {
                if (distractor_scenes[d] == query_item.s) {
                    continue;
                }
                float dist = metrics::l2Distance(query_desc, distractor_rows[d]);
                ranked.emplace_back(-dist, 0);
            }

            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            float ap = metrics::computeAPFromLabels(ranked, num_positives);
            sum_ap += ap;
            query_count++;
        }
    }
#else
    // Non-OpenMP fallback
    for (int q = 0; q < num_queries; ++q) {
        const auto& query_item = queries[q];
        if (!matchesSplit(query_item.s, split)) continue;

        const cv::Mat& query_scene_desc = bank.get(query_item.s, difficulty, "ref");
        if (query_scene_desc.empty() || query_item.idx >= query_scene_desc.rows) continue;

        cv::Mat query_desc = query_scene_desc.row(query_item.idx);

        std::string prefix = PatchKey::difficultyPrefix(difficulty);
        std::vector<cv::Mat> positive_descs;
        for (int i = 1; i <= 5; ++i) {
            std::string target = prefix + std::to_string(i);
            const cv::Mat& target_desc = bank.get(query_item.s, difficulty, target);
            if (!target_desc.empty() && query_item.idx < target_desc.rows) {
                positive_descs.push_back(target_desc.row(query_item.idx));
            }
        }

        if (positive_descs.empty()) continue;

        const int num_positives = static_cast<int>(positive_descs.size());

        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(positive_descs.size() + distractor_rows.size());

        for (const auto& pos_desc : positive_descs) {
            float dist = metrics::l2Distance(query_desc, pos_desc);
            ranked.emplace_back(-dist, 1);
        }

        for (size_t d = 0; d < distractor_rows.size(); ++d) {
            if (distractor_scenes[d] == query_item.s) {
                continue;
            }
            float dist = metrics::l2Distance(query_desc, distractor_rows[d]);
            ranked.emplace_back(-dist, 0);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float ap = metrics::computeAPFromLabels(ranked, num_positives);
        sum_ap += ap;
        query_count++;
    }
#endif

    if (query_count == 0) {
        return 0.0f;
    }

    return static_cast<float>(sum_ap / static_cast<double>(query_count));
}

bool RetrievalTask::matchesSplit(const std::string& scene, const std::string& split) {
    if (split == "full" || split == "all") {
        return true;
    }
    if (split == "illum") {
        return scene.size() > 2 && scene[0] == 'i' && scene[1] == '_';
    }
    if (split == "view") {
        return scene.size() > 2 && scene[0] == 'v' && scene[1] == '_';
    }
    return true;  // Unknown split, include all
}

} // namespace thesis_project::benchmark::tasks
