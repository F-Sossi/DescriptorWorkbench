#include "HPatchesBenchmark.hpp"
#include "core/patches/PatchLoader.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <random>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <filesystem>
#include <optional>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace thesis_project::benchmark {

using namespace patches;

namespace {

float computeAPFromLabels(const std::vector<std::pair<float, int>>& ranked, int positives);
float computeAPWithIgnore(const std::vector<std::pair<float, int>>& ranked, int positives);
float computeAPTrapz(const std::vector<float>& scores, const std::vector<int>& labels, int numpos = -1);
float l2Distance(const cv::Mat& a, const cv::Mat& b);
std::vector<float> computeRowDistancesSquared(const cv::Mat& a, const cv::Mat& b);

struct MapAccumulator {
    double sum_map = 0.0;
    int count = 0;

    void add(float ap) {
        sum_map += ap;
        count++;
    }

    float meanMAP() const {
        return count > 0 ? static_cast<float>(sum_map / count) : 0.0f;
    }
};

struct TaskAccumulators {
    MapAccumulator overall;
    MapAccumulator easy;
    MapAccumulator hard;
    MapAccumulator tough;
    MapAccumulator illumination;
    MapAccumulator viewpoint;
    MapAccumulator illumination_easy;
    MapAccumulator illumination_hard;
    MapAccumulator viewpoint_easy;
    MapAccumulator viewpoint_hard;
};

struct DescriptorCacheEntry {
    cv::Mat ref;
    cv::Mat target;
};

struct SceneKeyInfo {
    std::string scene_dir;
    std::string target_key;
    int num_patches = 0;
};

class DescriptorCache {
public:
    explicit DescriptorCache(size_t max_entries) : max_entries_(max_entries) {}

    DescriptorCacheEntry getOrLoad(const std::string& key,
                                   const std::string& scene_dir,
                                   const std::string& difficulty,
                                   const std::string& target_key,
                                   const HPatchesBenchmark::Config& config,
                                   patches::IPatchDescriptorExtractor& extractor,
                                   const DescriptorParams& params) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        auto scene = PatchLoader::loadScene(scene_dir, config.color);
        const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
        if (difficulty == "easy") {
            target_sets = &scene.easy;
        } else if (difficulty == "hard") {
            target_sets = &scene.hard;
        } else if (difficulty == "tough") {
            target_sets = &scene.tough;
        }

        if (!target_sets || target_sets->empty()) {
            return {};
        }

        auto it_target = target_sets->find(target_key);
        if (it_target == target_sets->end()) {
            return {};
        }

        DescriptorCacheEntry entry;
        entry.ref = extractor.extractFromPatches(scene.ref.patches, params);
        entry.target = extractor.extractFromPatches(it_target->second.patches, params);

        cache_[key] = entry;
        order_.push_back(key);
        if (order_.size() > max_entries_) {
            cache_.erase(order_.front());
            order_.erase(order_.begin());
        }

        return entry;
    }

private:
    size_t max_entries_;
    std::unordered_map<std::string, DescriptorCacheEntry> cache_;
    std::vector<std::string> order_;
    std::mutex mutex_;
};

struct PreloadedData {
    std::unordered_map<std::string, DescriptorCacheEntry> cache;
    std::unordered_map<std::string, std::vector<SceneKeyInfo>> entries;
    std::unordered_map<std::string, std::vector<std::string>> keys_by_scene;
    std::unordered_map<std::string, cv::Mat> diff_pools;
    std::unordered_map<std::string, cv::Mat> ref_cache;
};

struct PreloadPlan {
    std::unordered_map<std::string, std::unordered_set<std::string>> required_targets;
    std::unordered_set<std::string> required_refs;
};

struct DescriptorStore {
    database::DatabaseManager* db = nullptr;
    int descriptor_set_id = -1;
    bool read = false;
    bool write = false;
    std::unordered_map<std::string, cv::Mat> memo;
    std::mutex mutex;

    bool enabled() const {
        return db && descriptor_set_id >= 0 && (read || write);
    }
};

std::string splitKey(const std::string& split, const std::string& difficulty) {
    return split + "|" + difficulty;
}

std::string sceneKey(const std::string& scene, const std::string& difficulty, const std::string& target_key) {
    return scene + "|" + difficulty + "|" + target_key;
}

std::string sceneDifficultyKey(const std::string& scene, const std::string& difficulty) {
    return scene + "|" + difficulty;
}

std::string resolveSceneDir(const std::string& scene_name, const HPatchesBenchmark::Config& config);
std::string targetKeyForIndex(const std::string& difficulty, int t);

std::string descriptorKey(const std::string& scene, const std::string& difficulty, const std::string& target_key) {
    return scene + "|" + difficulty + "|" + target_key;
}

void addRequiredTarget(PreloadPlan& plan, const std::string& scene_dir, const std::string& difficulty,
                       const std::string& target_key) {
    if (target_key.empty()) {
        return;
    }
    const std::string ref_key = sceneDifficultyKey(scene_dir, difficulty);
    plan.required_targets[ref_key].insert(target_key);
}

void addRequiredRef(PreloadPlan& plan, const std::string& scene_dir, const std::string& difficulty) {
    plan.required_refs.insert(sceneDifficultyKey(scene_dir, difficulty));
}

PreloadPlan buildPreloadPlanFromTasks(const HPatchesBenchmark::Config& config) {
    PreloadPlan plan;
    const std::vector<std::string> difficulties = {"easy", "hard", "tough"};

    auto addPair = [&](const HPatchesBenchmark::VerificationTaskPair& pair) {
        const std::string scene1 = resolveSceneDir(pair.s1, config);
        const std::string scene2 = resolveSceneDir(pair.s2, config);
        for (const auto& difficulty : difficulties) {
            if ((difficulty == "easy" && !config.include_easy) ||
                (difficulty == "hard" && !config.include_hard) ||
                (difficulty == "tough" && !config.include_tough)) {
                continue;
            }
            if (pair.t1 == 0) {
                addRequiredRef(plan, scene1, difficulty);
            } else {
                addRequiredTarget(plan, scene1, difficulty, targetKeyForIndex(difficulty, pair.t1));
            }
            if (pair.t2 == 0) {
                addRequiredRef(plan, scene2, difficulty);
            } else {
                addRequiredTarget(plan, scene2, difficulty, targetKeyForIndex(difficulty, pair.t2));
            }
        }
    };

    for (const auto& pair : config.tasks.verification_pos_pairs) {
        addPair(pair);
    }
    for (const auto& pair : config.tasks.verification_neg_inter_pairs) {
        addPair(pair);
    }
    for (const auto& pair : config.tasks.verification_neg_intra_pairs) {
        addPair(pair);
    }

    for (const auto& item : config.tasks.retrieval_queries) {
        const std::string scene = resolveSceneDir(item.s, config);
        for (const auto& difficulty : difficulties) {
            if ((difficulty == "easy" && !config.include_easy) ||
                (difficulty == "hard" && !config.include_hard) ||
                (difficulty == "tough" && !config.include_tough)) {
                continue;
            }
            addRequiredRef(plan, scene, difficulty);
            for (int t = 1; t <= 5; ++t) {
                addRequiredTarget(plan, scene, difficulty, targetKeyForIndex(difficulty, t));
            }
        }
    }

    for (const auto& item : config.tasks.retrieval_distractors) {
        const std::string scene = resolveSceneDir(item.s, config);
        for (const auto& difficulty : difficulties) {
            if ((difficulty == "easy" && !config.include_easy) ||
                (difficulty == "hard" && !config.include_hard) ||
                (difficulty == "tough" && !config.include_tough)) {
                continue;
            }
            addRequiredRef(plan, scene, difficulty);
        }
    }

    return plan;
}

PreloadedData preloadDescriptors(
    const std::vector<std::string>& scene_dirs,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    const PreloadPlan* plan,
    DescriptorStore* store) {

    PreloadedData data;
    const std::vector<std::string> difficulties = {"easy", "hard", "tough"};

    for (const auto& scene_dir : scene_dirs) {
        const bool is_illum = PatchLoader::isIlluminationScene(scene_dir);
        const std::string split = is_illum ? "illum" : "view";
        for (const auto& difficulty : difficulties) {
            if ((difficulty == "easy" && !config.include_easy) ||
                (difficulty == "hard" && !config.include_hard) ||
                (difficulty == "tough" && !config.include_tough)) {
                continue;
            }

            const std::string ref_key = sceneDifficultyKey(scene_dir, difficulty);
            const bool need_ref = plan && plan->required_refs.find(ref_key) != plan->required_refs.end();
            const bool need_targets = plan && plan->required_targets.find(ref_key) != plan->required_targets.end();
            if (plan && !need_ref && !need_targets) {
                continue;
            }

            if (!need_targets) {
                if (store && store->enabled() && store->read) {
                    const std::string dkey = descriptorKey(scene_dir, difficulty, "ref");
                    {
                        std::lock_guard<std::mutex> lock(store->mutex);
                        auto memo_it = store->memo.find(dkey);
                        if (memo_it != store->memo.end()) {
                            data.ref_cache.emplace(ref_key, memo_it->second);
                            continue;
                        }
                        auto db_mat = store->db->loadPatchBenchmarkDescriptor(
                            store->descriptor_set_id, std::filesystem::path(scene_dir).filename().string(),
                            difficulty, "ref");
                        if (db_mat) {
                            store->memo.emplace(dkey, *db_mat);
                            data.ref_cache.emplace(ref_key, *db_mat);
                            continue;
                        }
                    }
                }
                auto scene = PatchLoader::loadScene(scene_dir, config.color);
                DescriptorCacheEntry entry;
                entry.ref = extractor.extractFromPatches(scene.ref.patches, params);
                if (!entry.ref.empty()) {
                    data.ref_cache.emplace(ref_key, entry.ref);
                    if (store && store->enabled() && store->write) {
                        std::lock_guard<std::mutex> lock(store->mutex);
                        store->db->storePatchBenchmarkDescriptor(
                            store->descriptor_set_id, std::filesystem::path(scene_dir).filename().string(),
                            difficulty, "ref", entry.ref);
                        store->memo.emplace(descriptorKey(scene_dir, difficulty, "ref"), entry.ref);
                    }
                }
                continue;
            }

            auto scene = PatchLoader::loadScene(scene_dir, config.color);
            const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
            if (difficulty == "easy") target_sets = &scene.easy;
            else if (difficulty == "hard") target_sets = &scene.hard;
            else if (difficulty == "tough") target_sets = &scene.tough;

            if (!target_sets || target_sets->empty()) {
                continue;
            }

            const std::unordered_set<std::string>* required_targets = nullptr;
            if (plan) {
                auto it = plan->required_targets.find(ref_key);
                if (it != plan->required_targets.end()) {
                    required_targets = &it->second;
                }
            }

            for (const auto& [key, patch_set] : *target_sets) {
                if (required_targets && required_targets->find(key) == required_targets->end()) {
                    continue;
                }
                if (patch_set.patches.size() != scene.ref.patches.size()) {
                    continue;
                }

                DescriptorCacheEntry entry;
                const std::string scene_label = std::filesystem::path(scene_dir).filename().string();
                const std::string target_key = key;
                if (store && store->enabled() && store->read) {
                    const std::string ref_dkey = descriptorKey(scene_dir, difficulty, "ref");
                    const std::string tgt_dkey = descriptorKey(scene_dir, difficulty, target_key);
                    std::lock_guard<std::mutex> lock(store->mutex);
                    auto ref_it = store->memo.find(ref_dkey);
                    if (ref_it != store->memo.end()) {
                        entry.ref = ref_it->second;
                    } else {
                        auto db_mat = store->db->loadPatchBenchmarkDescriptor(
                            store->descriptor_set_id, scene_label, difficulty, "ref");
                        if (db_mat) {
                            store->memo.emplace(ref_dkey, *db_mat);
                            entry.ref = *db_mat;
                        }
                    }
                    auto tgt_it = store->memo.find(tgt_dkey);
                    if (tgt_it != store->memo.end()) {
                        entry.target = tgt_it->second;
                    } else {
                        auto db_mat = store->db->loadPatchBenchmarkDescriptor(
                            store->descriptor_set_id, scene_label, difficulty, target_key);
                        if (db_mat) {
                            store->memo.emplace(tgt_dkey, *db_mat);
                            entry.target = *db_mat;
                        }
                    }
                }
                if (entry.ref.empty()) {
                    entry.ref = extractor.extractFromPatches(scene.ref.patches, params);
                }
                if (entry.target.empty()) {
                    entry.target = extractor.extractFromPatches(patch_set.patches, params);
                }
                if (store && store->enabled() && store->write) {
                    std::lock_guard<std::mutex> lock(store->mutex);
                    if (!entry.ref.empty()) {
                        store->db->storePatchBenchmarkDescriptor(
                            store->descriptor_set_id, scene_label, difficulty, "ref", entry.ref);
                        store->memo.emplace(descriptorKey(scene_dir, difficulty, "ref"), entry.ref);
                    }
                    if (!entry.target.empty()) {
                        store->db->storePatchBenchmarkDescriptor(
                            store->descriptor_set_id, scene_label, difficulty, target_key, entry.target);
                        store->memo.emplace(descriptorKey(scene_dir, difficulty, target_key), entry.target);
                    }
                }

                const std::string cache_key = sceneKey(scene_dir, difficulty, key);
                data.cache.emplace(cache_key, entry);
                data.entries[splitKey("all", difficulty)].push_back({scene_dir, key, static_cast<int>(scene.ref.patches.size())});
                data.entries[splitKey(split, difficulty)].push_back({scene_dir, key, static_cast<int>(scene.ref.patches.size())});
                data.keys_by_scene[sceneDifficultyKey(scene_dir, difficulty)].push_back(key);
                const std::string ref_key = sceneDifficultyKey(scene_dir, difficulty);
                if (data.ref_cache.find(ref_key) == data.ref_cache.end()) {
                    data.ref_cache.emplace(ref_key, entry.ref);
                }
            }
        }
    }

    return data;
}

std::string resolveSceneDir(const std::string& scene_name, const HPatchesBenchmark::Config& config) {
    if (scene_name.empty()) {
        return scene_name;
    }
    if (scene_name.find('/') != std::string::npos) {
        return scene_name;
    }
    return (std::filesystem::path(config.patches_dir) / scene_name).string();
}

std::string targetKeyForIndex(const std::string& difficulty, int t) {
    if (t <= 0 || t > 5) {
        return {};
    }
    if (difficulty == "easy") {
        return "e" + std::to_string(t);
    }
    if (difficulty == "hard") {
        return "h" + std::to_string(t);
    }
    if (difficulty == "tough") {
        return "t" + std::to_string(t);
    }
    return {};
}

cv::Mat getDescriptorsForImage(
    const std::string& scene_name,
    const std::string& difficulty,
    int t,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    PreloadedData* preloaded,
    DescriptorCache& cache,
    std::unordered_map<std::string, cv::Mat>& ref_cache,
    DescriptorStore* store) {

    const std::string scene_dir = resolveSceneDir(scene_name, config);
    if (scene_dir.empty()) {
        return {};
    }

    if (t == 0) {
        const std::string ref_key = sceneDifficultyKey(scene_dir, difficulty);
        if (preloaded) {
            auto it = preloaded->ref_cache.find(ref_key);
            if (it != preloaded->ref_cache.end()) {
                return it->second;
            }
        }
        auto it = ref_cache.find(ref_key);
        if (it != ref_cache.end()) {
            return it->second;
        }
        if (store && store->enabled() && store->read) {
            const std::string dkey = descriptorKey(scene_dir, difficulty, "ref");
            {
                std::lock_guard<std::mutex> lock(store->mutex);
                auto memo_it = store->memo.find(dkey);
                if (memo_it != store->memo.end()) {
                    ref_cache.emplace(ref_key, memo_it->second);
                    return memo_it->second;
                }
                auto db_mat = store->db->loadPatchBenchmarkDescriptor(
                    store->descriptor_set_id, scene_name, difficulty, "ref");
                if (db_mat) {
                    store->memo.emplace(dkey, *db_mat);
                    ref_cache.emplace(ref_key, *db_mat);
                    return *db_mat;
                }
            }
        }
        auto scene = PatchLoader::loadScene(scene_dir, config.color);
        cv::Mat ref_desc = extractor.extractFromPatches(scene.ref.patches, params);
        ref_cache.emplace(ref_key, ref_desc);
        if (store && store->enabled() && store->write && !ref_desc.empty()) {
            std::lock_guard<std::mutex> lock(store->mutex);
            store->db->storePatchBenchmarkDescriptor(
                store->descriptor_set_id, scene_name, difficulty, "ref", ref_desc);
            store->memo.emplace(descriptorKey(scene_dir, difficulty, "ref"), ref_desc);
        }
        return ref_desc;
    }

    const std::string key = targetKeyForIndex(difficulty, t);
    if (key.empty()) {
        return {};
    }
    const std::string cache_key = sceneKey(scene_dir, difficulty, key);
    if (preloaded) {
        auto it = preloaded->cache.find(cache_key);
        if (it != preloaded->cache.end()) {
            return it->second.target;
        }
        return {};
    }

    if (store && store->enabled() && store->read) {
        const std::string dkey = descriptorKey(scene_dir, difficulty, key);
        {
            std::lock_guard<std::mutex> lock(store->mutex);
            auto memo_it = store->memo.find(dkey);
            if (memo_it != store->memo.end()) {
                return memo_it->second;
            }
            auto db_mat = store->db->loadPatchBenchmarkDescriptor(
                store->descriptor_set_id, scene_name, difficulty, key);
            if (db_mat) {
                store->memo.emplace(dkey, *db_mat);
                return *db_mat;
            }
        }
    }

    DescriptorCacheEntry descs = cache.getOrLoad(cache_key, scene_dir, difficulty, key, config, extractor, params);
    if (store && store->enabled() && store->write && !descs.target.empty()) {
        std::lock_guard<std::mutex> lock(store->mutex);
        store->db->storePatchBenchmarkDescriptor(
            store->descriptor_set_id, scene_name, difficulty, key, descs.target);
        store->memo.emplace(descriptorKey(scene_dir, difficulty, key), descs.target);
    }
    return descs.target;
}

bool passSplitFilter(const std::string& scene_name, const std::string& split) {
    if (split == "illum") {
        return PatchLoader::isIlluminationScene(scene_name);
    }
    if (split == "view") {
        return PatchLoader::isViewpointScene(scene_name);
    }
    return true;
}

float computeVerificationFromTasks(
    const std::vector<HPatchesBenchmark::VerificationTaskPair>& pos_pairs,
    const std::vector<HPatchesBenchmark::VerificationTaskPair>& neg_pairs,
    const std::string& difficulty,
    const std::string& split,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    PreloadedData* preloaded,
    DescriptorStore* store) {

    if (pos_pairs.empty()) {
        return 0.0f;
    }

    DescriptorCache cache(32);
    std::unordered_map<std::string, cv::Mat> ref_cache;

    std::vector<float> pos_scores;
    std::vector<float> neg_scores;
    pos_scores.reserve(pos_pairs.size());
    neg_scores.reserve(neg_pairs.size());

    for (const auto& pair : pos_pairs) {
        if (!passSplitFilter(pair.s1, split) || !passSplitFilter(pair.s2, split)) {
            continue;
        }
        cv::Mat desc1 = getDescriptorsForImage(pair.s1, difficulty, pair.t1, config, extractor, params,
                                               preloaded, cache, ref_cache, store);
        cv::Mat desc2 = getDescriptorsForImage(pair.s2, difficulty, pair.t2, config, extractor, params,
                                               preloaded, cache, ref_cache, store);
        if (desc1.empty() || desc2.empty()) {
            continue;
        }
        if (pair.idx1 < 0 || pair.idx1 >= desc1.rows || pair.idx2 < 0 || pair.idx2 >= desc2.rows) {
            continue;
        }
        const float dist = l2Distance(desc1.row(pair.idx1), desc2.row(pair.idx2));
        pos_scores.push_back(-dist);
    }

    for (const auto& pair : neg_pairs) {
        if (!passSplitFilter(pair.s1, split) || !passSplitFilter(pair.s2, split)) {
            continue;
        }
        cv::Mat desc1 = getDescriptorsForImage(pair.s1, difficulty, pair.t1, config, extractor, params,
                                               preloaded, cache, ref_cache, store);
        cv::Mat desc2 = getDescriptorsForImage(pair.s2, difficulty, pair.t2, config, extractor, params,
                                               preloaded, cache, ref_cache, store);
        if (desc1.empty() || desc2.empty()) {
            continue;
        }
        if (pair.idx1 < 0 || pair.idx1 >= desc1.rows || pair.idx2 < 0 || pair.idx2 >= desc2.rows) {
            continue;
        }
        const float dist = l2Distance(desc1.row(pair.idx1), desc2.row(pair.idx2));
        neg_scores.push_back(-dist);
    }

    if (pos_scores.empty() || neg_scores.empty()) {
        return 0.0f;
    }

    const size_t pos_total = pos_scores.size();
    const size_t pos_subset = static_cast<size_t>(std::max(1.0, std::floor(pos_total * 0.2)));
    const size_t pos_take = std::min(pos_total, pos_subset);

    std::vector<float> scores;
    std::vector<int> labels;
    scores.reserve(neg_scores.size() + pos_take);
    labels.reserve(neg_scores.size() + pos_take);

    for (float score : neg_scores) {
        scores.push_back(score);
        labels.push_back(0);
    }
    for (size_t i = 0; i < pos_take; ++i) {
        scores.push_back(pos_scores[i]);
        labels.push_back(1);
    }

    return computeAPTrapz(scores, labels);
}

float computeRetrievalFromTasks(
    const std::vector<HPatchesBenchmark::RetrievalTaskItem>& queries,
    const std::vector<HPatchesBenchmark::RetrievalTaskItem>& distractors,
    const std::string& difficulty,
    const std::string& split,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    PreloadedData* preloaded,
    DescriptorStore* store) {

    if (queries.empty()) {
        return 0.0f;
    }

    DescriptorCache cache(32);
    std::unordered_map<std::string, cv::Mat> ref_cache;

    struct DistractorRow {
        cv::Mat desc;
        std::string scene;
    };
    std::vector<DistractorRow> distractor_rows;
    distractor_rows.reserve(distractors.size());
    for (const auto& item : distractors) {
        if (!passSplitFilter(item.s, split)) {
            continue;
        }
        cv::Mat desc = getDescriptorsForImage(item.s, difficulty, 0, config, extractor, params,
                                              preloaded, cache, ref_cache, store);
        if (desc.empty() || item.idx < 0 || item.idx >= desc.rows) {
            continue;
        }
        distractor_rows.push_back({desc.row(item.idx), item.s});
    }

    double sum_ap = 0.0;
    int count = 0;
    for (const auto& query : queries) {
        if (!passSplitFilter(query.s, split)) {
            continue;
        }
        cv::Mat ref_desc = getDescriptorsForImage(query.s, difficulty, 0, config, extractor, params,
                                                  preloaded, cache, ref_cache, store);
        if (ref_desc.empty() || query.idx < 0 || query.idx >= ref_desc.rows) {
            continue;
        }
        std::vector<std::pair<float, int>> ranked;
        int positives = 0;
        for (int t = 1; t <= 5; ++t) {
            cv::Mat tgt_desc = getDescriptorsForImage(query.s, difficulty, t, config, extractor, params,
                                                      preloaded, cache, ref_cache, store);
            if (tgt_desc.empty() || query.idx >= tgt_desc.rows) {
                continue;
            }
            const float dist = l2Distance(ref_desc.row(query.idx), tgt_desc.row(query.idx));
            ranked.emplace_back(-dist, 1);
            positives++;
        }
        if (positives == 0) {
            continue;
        }

        for (const auto& row : distractor_rows) {
            const float dist = l2Distance(ref_desc.row(query.idx), row.desc);
            const int label = (row.scene == query.s) ? 0 : -1;
            ranked.emplace_back(-dist, label);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        sum_ap += computeAPWithIgnore(ranked, positives);
        count++;
    }

    return count > 0 ? static_cast<float>(sum_ap / static_cast<double>(count)) : 0.0f;
}

std::vector<SceneKeyInfo> buildSceneKeyInfo(
    const std::vector<std::string>& scenes,
    const std::string& difficulty,
    const HPatchesBenchmark::Config& config) {

    std::vector<SceneKeyInfo> info;
    for (const auto& scene_dir : scenes) {
        auto scene = PatchLoader::loadScene(scene_dir, config.color);
        const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
        if (difficulty == "easy") {
            target_sets = &scene.easy;
        } else if (difficulty == "hard") {
            target_sets = &scene.hard;
        } else if (difficulty == "tough") {
            target_sets = &scene.tough;
        }
        if (!target_sets || target_sets->empty()) {
            continue;
        }
        for (const auto& [key, patch_set] : *target_sets) {
            if (patch_set.patches.size() != scene.ref.patches.size()) {
                continue;
            }
            SceneKeyInfo entry;
            entry.scene_dir = scene_dir;
            entry.target_key = key;
            entry.num_patches = static_cast<int>(scene.ref.patches.size());
            info.push_back(entry);
        }
    }
    return info;
}

float computeVerificationGlobal(
    const std::vector<std::string>& scenes,
    const std::string& difficulty,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    bool diff_seq,
    PreloadedData* preloaded,
    const std::string& split) {

    auto entries = preloaded
        ? preloaded->entries[splitKey(split, difficulty)]
        : buildSceneKeyInfo(scenes, difficulty, config);
    if (entries.empty()) {
        return 0.0f;
    }

    const int positives = config.tasks.verification_num_positives;
    const int negatives = config.tasks.verification_num_negatives;

    std::mt19937 rng(config.tasks.random_seed +
                     static_cast<unsigned int>(std::hash<std::string>{}(difficulty + (diff_seq ? "diff" : "same"))));

    std::uniform_int_distribution<int> entry_dist(0, static_cast<int>(entries.size() - 1));

    DescriptorCache cache(12);
    std::vector<float> pos_scores;
    pos_scores.reserve(static_cast<size_t>(positives));

    for (int p = 0; p < positives; ++p) {
        const auto& entry = entries[entry_dist(rng)];
        const std::string cache_key = sceneKey(entry.scene_dir, difficulty, entry.target_key);
        DescriptorCacheEntry descs = preloaded
            ? preloaded->cache.at(cache_key)
            : cache.getOrLoad(cache_key, entry.scene_dir, difficulty, entry.target_key, config, extractor, params);
        if (descs.ref.empty() || descs.target.empty()) {
            continue;
        }
        std::uniform_int_distribution<int> idx_dist(0, entry.num_patches - 1);
        int idx = idx_dist(rng);
        const float dist = l2Distance(descs.ref.row(idx), descs.target.row(idx));
        pos_scores.push_back(-dist);
    }

    std::vector<float> neg_scores;
    neg_scores.reserve(static_cast<size_t>(negatives));

    // Reuse a fixed diff-seq pool per split
    cv::Mat diff_pool;
    if (diff_seq) {
        const std::string pool_key = splitKey(split, difficulty) + "|diff_pool|" + std::to_string(negatives);
        static std::unordered_map<std::string, cv::Mat> pool_cache;
        static std::mutex pool_mutex;
        auto it = pool_cache.find(pool_key);
        if (it != pool_cache.end()) {
            diff_pool = it->second;
        } else {
            std::lock_guard<std::mutex> lock(pool_mutex);
            auto it2 = pool_cache.find(pool_key);
            if (it2 != pool_cache.end()) {
                diff_pool = it2->second;
            } else {
            std::vector<cv::Mat> pool_rows;
            pool_rows.reserve(static_cast<size_t>(negatives));
            while (static_cast<int>(pool_rows.size()) < negatives) {
                const auto& entry = entries[entry_dist(rng)];
                const std::string cache_key = sceneKey(entry.scene_dir, difficulty, entry.target_key);
                DescriptorCacheEntry descs = preloaded
                    ? preloaded->cache.at(cache_key)
                    : cache.getOrLoad(cache_key, entry.scene_dir, difficulty, entry.target_key, config, extractor, params);
                if (descs.target.empty()) {
                    continue;
                }
                std::uniform_int_distribution<int> idx_dist(0, entry.num_patches - 1);
                int idx = idx_dist(rng);
                pool_rows.push_back(descs.target.row(idx));
            }
            if (!pool_rows.empty()) {
                diff_pool = cv::Mat(static_cast<int>(pool_rows.size()),
                                    pool_rows.front().cols,
                                    pool_rows.front().type());
                for (int i = 0; i < diff_pool.rows; ++i) {
                    pool_rows[i].copyTo(diff_pool.row(i));
                }
                if (preloaded) {
                    preloaded->diff_pools[pool_key] = diff_pool;
                }
                pool_cache[pool_key] = diff_pool;
            }
            }
        }
    }

    for (int n = 0; n < negatives; ++n) {
        const auto& entry = entries[entry_dist(rng)];
        const std::string cache_key = sceneKey(entry.scene_dir, difficulty, entry.target_key);
        DescriptorCacheEntry descs = preloaded
            ? preloaded->cache.at(cache_key)
            : cache.getOrLoad(cache_key, entry.scene_dir, difficulty, entry.target_key, config, extractor, params);
        if (descs.ref.empty() || descs.target.empty()) {
            continue;
        }
        std::uniform_int_distribution<int> idx_dist(0, entry.num_patches - 1);
        int idx_ref = idx_dist(rng);
        if (!diff_seq) {
            int idx_tgt = idx_dist(rng);
            if (idx_tgt == idx_ref) {
                idx_tgt = (idx_tgt + 1) % entry.num_patches;
            }
            const float dist = l2Distance(descs.ref.row(idx_ref), descs.target.row(idx_tgt));
            neg_scores.push_back(-dist);
        } else if (!diff_pool.empty()) {
            std::uniform_int_distribution<int> pool_dist(0, diff_pool.rows - 1);
            int idx_neg = pool_dist(rng);
            const float dist = l2Distance(descs.ref.row(idx_ref), diff_pool.row(idx_neg));
            neg_scores.push_back(-dist);
        }
    }

    if (pos_scores.empty() || neg_scores.empty()) {
        return 0.0f;
    }

    const size_t pos_total = pos_scores.size();
    const size_t pos_subset = static_cast<size_t>(std::max(1.0, std::floor(pos_total * 0.2)));
    const size_t pos_take = std::min(pos_total, pos_subset);

    std::vector<float> scores;
    std::vector<int> labels;
    scores.reserve(neg_scores.size() + pos_take);
    labels.reserve(neg_scores.size() + pos_take);
    for (float score : neg_scores) {
        scores.push_back(score);
        labels.push_back(0);
    }
    for (size_t i = 0; i < pos_take; ++i) {
        scores.push_back(pos_scores[i]);
        labels.push_back(1);
    }

    return computeAPTrapz(scores, labels);
}

float computeRetrievalGlobal(
    const std::vector<std::string>& scenes,
    const std::string& difficulty,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    PreloadedData* preloaded,
    const std::string& split) {

    auto entries = preloaded
        ? preloaded->entries[splitKey(split, difficulty)]
        : buildSceneKeyInfo(scenes, difficulty, config);
    if (entries.empty()) {
        return 0.0f;
    }

    std::mt19937 rng(config.tasks.random_seed +
                     static_cast<unsigned int>(std::hash<std::string>{}(difficulty + "retrieval")));
    std::uniform_int_distribution<int> entry_dist(0, static_cast<int>(entries.size() - 1));

    DescriptorCache cache(12);

    // Build distractor pool
    std::vector<cv::Mat> distractor_rows;
    distractor_rows.reserve(static_cast<size_t>(config.tasks.retrieval_num_distractors));
    while (static_cast<int>(distractor_rows.size()) < config.tasks.retrieval_num_distractors) {
        const auto& entry = entries[entry_dist(rng)];
        const std::string cache_key = sceneKey(entry.scene_dir, difficulty, entry.target_key);
        DescriptorCacheEntry descs = preloaded
            ? preloaded->cache.at(cache_key)
            : cache.getOrLoad(cache_key, entry.scene_dir, difficulty, entry.target_key, config, extractor, params);
        if (descs.target.empty()) {
            continue;
        }
        std::uniform_int_distribution<int> idx_dist(0, entry.num_patches - 1);
        int idx = idx_dist(rng);
        distractor_rows.push_back(descs.target.row(idx));
    }
    cv::Mat distractors;
    if (!distractor_rows.empty()) {
        distractors = cv::Mat(static_cast<int>(distractor_rows.size()),
                              distractor_rows.front().cols,
                              distractor_rows.front().type());
        for (int i = 0; i < distractors.rows; ++i) {
            distractor_rows[i].copyTo(distractors.row(i));
        }
    }

    const int queries_target = config.tasks.retrieval_num_queries;
    std::vector<SceneKeyInfo> query_entries;
    query_entries.reserve(static_cast<size_t>(queries_target));
    while (static_cast<int>(query_entries.size()) < queries_target) {
        query_entries.push_back(entries[entry_dist(rng)]);
    }

    double sum_ap = 0.0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum_ap)
    #endif
    for (int qi_idx = 0; qi_idx < static_cast<int>(query_entries.size()); ++qi_idx) {
        const auto& entry = query_entries[qi_idx];
        std::mt19937 local_rng(config.tasks.random_seed + static_cast<unsigned int>(qi_idx));

        const std::string cache_key = sceneKey(entry.scene_dir, difficulty, entry.target_key);
        DescriptorCacheEntry descs = preloaded
            ? preloaded->cache.at(cache_key)
            : cache.getOrLoad(cache_key, entry.scene_dir, difficulty, entry.target_key, config, extractor, params);
        if (descs.ref.empty()) {
            continue;
        }

        std::uniform_int_distribution<int> idx_dist(0, entry.num_patches - 1);
        int qi = idx_dist(local_rng);

        std::vector<cv::Mat> target_descs;
        if (preloaded) {
            const auto& keys = preloaded->keys_by_scene[sceneDifficultyKey(entry.scene_dir, difficulty)];
            target_descs.reserve(keys.size());
            for (const auto& key : keys) {
                const std::string tkey = sceneKey(entry.scene_dir, difficulty, key);
                auto cached = preloaded->cache.at(tkey);
                if (!cached.target.empty()) {
                    target_descs.push_back(cached.target);
                }
            }
        } else {
            auto scene = PatchLoader::loadScene(entry.scene_dir, config.color);
            const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
            if (difficulty == "easy") {
                target_sets = &scene.easy;
            } else if (difficulty == "hard") {
                target_sets = &scene.hard;
            } else if (difficulty == "tough") {
                target_sets = &scene.tough;
            }
            if (!target_sets || target_sets->empty()) {
                continue;
            }
            target_descs.reserve(target_sets->size());
            for (const auto& [key, patch_set] : *target_sets) {
                if (patch_set.patches.size() != scene.ref.patches.size()) {
                    continue;
                }
                const std::string tkey = sceneKey(entry.scene_dir, difficulty, key);
                auto cached = cache.getOrLoad(tkey, entry.scene_dir, difficulty, key, config, extractor, params);
                if (!cached.target.empty()) {
                    target_descs.push_back(cached.target);
                }
            }
        }

        if (target_descs.empty()) {
            continue;
        }

        int positives = static_cast<int>(target_descs.size());
        std::vector<std::pair<float, int>> ranked;
        for (const auto& target_desc : target_descs) {
            for (int j = 0; j < target_desc.rows; ++j) {
                const float dist = l2Distance(descs.ref.row(qi), target_desc.row(j));
                int label = (j == qi) ? 1 : 0;
                ranked.emplace_back(-dist, label);
            }
        }
        for (int j = 0; j < distractors.rows; ++j) {
            const float dist = l2Distance(descs.ref.row(qi), distractors.row(j));
            ranked.emplace_back(-dist, -1);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        sum_ap += computeAPWithIgnore(ranked, positives);
    }

    return static_cast<float>(sum_ap / static_cast<double>(queries_target));
}

float computeAPFromLabels(const std::vector<std::pair<float, int>>& ranked, int positives) {
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

float computeAPTrapz(const std::vector<float>& scores,
                     const std::vector<int>& labels,
                     int numpos) {
    if (scores.empty() || labels.empty() || scores.size() != labels.size()) {
        return 0.0f;
    }

    int pos_count = 0;
    for (int label : labels) {
        if (label == 1) {
            pos_count++;
        }
    }
    if (pos_count == 0) {
        return 0.0f;
    }

    std::vector<float> work_scores = scores;
    std::vector<int> work_labels = labels;
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

    std::vector<int> perm(work_scores.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](int a, int b) { return work_scores[a] > work_scores[b]; });

    int last_valid = -1;
    for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
        if (work_scores[perm[i]] > -std::numeric_limits<float>::infinity()) {
            last_valid = i;
        }
    }
    if (last_valid < 0) {
        return 0.0f;
    }

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

    std::vector<float> recall(tp.size());
    std::vector<float> precision(tp.size());
    const float denom_pos = std::max(1.0f, static_cast<float>(pos_count));
    for (size_t i = 0; i < tp.size(); ++i) {
        recall[i] = tp[i] / denom_pos;
        const float denom = std::max(1e-10f, tp[i] + fp[i]);
        precision[i] = tp[i] / denom;
    }

    float ap = 0.0f;
    for (size_t i = 1; i < recall.size(); ++i) {
        const float dr = recall[i] - recall[i - 1];
        ap += dr * (precision[i] + precision[i - 1]) * 0.5f;
    }
    return ap;
}
float computeAPWithIgnore(const std::vector<std::pair<float, int>>& ranked, int positives) {
    if (positives <= 0 || ranked.empty()) {
        return 0.0f;
    }
    int hits = 0;
    int retrieved = 0;
    double sum_prec = 0.0;
    for (const auto& [score, label] : ranked) {
        if (label == 0) {
            continue;
        }
        retrieved++;
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

float l2Distance(const cv::Mat& a, const cv::Mat& b) {
    return static_cast<float>(cv::norm(a, b, cv::NORM_L2));
}

std::vector<float> computeRowDistancesSquared(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.rows != b.rows || a.cols != b.cols) {
        return {};
    }
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

cv::Mat buildDiffSeqPool(
    const std::string& current_scene,
    const std::string& difficulty,
    const std::string& key,
    int pool_size,
    bool is_illumination,
    const std::vector<std::string>& illumination_scenes,
    const std::vector<std::string>& viewpoint_scenes,
    const HPatchesBenchmark::Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    unsigned int seed) {

    if (pool_size <= 0) {
        return {};
    }

    std::vector<std::string> candidates = is_illumination ? illumination_scenes : viewpoint_scenes;
    candidates.erase(
        std::remove(candidates.begin(), candidates.end(), current_scene),
        candidates.end());

    std::mt19937 rng(seed);
    std::shuffle(candidates.begin(), candidates.end(), rng);

    std::vector<cv::Mat> patches;
    patches.reserve(pool_size);

    for (const auto& scene_dir : candidates) {
        if (static_cast<int>(patches.size()) >= pool_size) {
            break;
        }

        auto scene = PatchLoader::loadScene(scene_dir, config.color);
        const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
        if (difficulty == "easy") {
            target_sets = &scene.easy;
        } else if (difficulty == "hard") {
            target_sets = &scene.hard;
        } else if (difficulty == "tough") {
            target_sets = &scene.tough;
        }

        if (!target_sets) {
            continue;
        }

        auto it = target_sets->find(key);
        if (it == target_sets->end()) {
            continue;
        }

        const auto& set_patches = it->second.patches;
        if (set_patches.empty()) {
            continue;
        }

        const int remaining = pool_size - static_cast<int>(patches.size());
        auto sample = sampleIndices(static_cast<int>(set_patches.size()), remaining, rng);
        for (int idx : sample) {
            patches.push_back(set_patches[idx]);
        }
    }

    if (patches.empty()) {
        return {};
    }

    return extractor.extractFromPatches(patches, params);
}

float computeVerificationSameSeq(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    int negatives_per_query,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries) {
        return 0.0f;
    }

    const int negatives = std::min(negatives_per_query, num_queries - 1);
    if (negatives <= 0) {
        return 0.0f;
    }

    double sum_ap = 0.0;
    for (int i = 0; i < num_queries; ++i) {
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(static_cast<size_t>(negatives + 1));

        const float pos_dist = l2Distance(ref_desc.row(i), target_desc.row(i));
        ranked.emplace_back(pos_dist, 1);

        auto negatives_idx = sampleUniqueIndices(num_queries, i, negatives, rng);
        for (int idx : negatives_idx) {
            const float neg_dist = l2Distance(ref_desc.row(i), target_desc.row(idx));
            ranked.emplace_back(neg_dist, 0);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        sum_ap += computeAPFromLabels(ranked, 1);
    }

    return static_cast<float>(sum_ap / static_cast<double>(num_queries));
}

float computeVerificationDiffSeq(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const cv::Mat& negative_pool,
    int negatives_per_query,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries || negative_pool.rows == 0) {
        return 0.0f;
    }

    const int negatives = std::min(negatives_per_query, negative_pool.rows);
    if (negatives <= 0) {
        return 0.0f;
    }

    std::uniform_int_distribution<int> dist(0, negative_pool.rows - 1);
    double sum_ap = 0.0;
    for (int i = 0; i < num_queries; ++i) {
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(static_cast<size_t>(negatives + 1));

        const float pos_dist = l2Distance(ref_desc.row(i), target_desc.row(i));
        ranked.emplace_back(pos_dist, 1);

        for (int n = 0; n < negatives; ++n) {
            const int idx = dist(rng);
            const float neg_dist = l2Distance(ref_desc.row(i), negative_pool.row(idx));
            ranked.emplace_back(neg_dist, 0);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        sum_ap += computeAPFromLabels(ranked, 1);
    }

    return static_cast<float>(sum_ap / static_cast<double>(num_queries));
}

float computeVerificationSameSeqPaper(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    int positives_target,
    int negatives_target,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries) {
        return 0.0f;
    }

    const int positives = std::min(positives_target, num_queries);
    std::vector<int> pos_indices = sampleIndices(num_queries, positives, rng);

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(positives + negatives_target));

    for (int idx : pos_indices) {
        const float dist = l2Distance(ref_desc.row(idx), target_desc.row(idx));
        ranked.emplace_back(-dist, 1);
    }

    const int max_neg = std::min(negatives_target, num_queries * (num_queries - 1));
    std::uniform_int_distribution<int> dist_idx(0, num_queries - 1);
    for (int n = 0; n < max_neg; ++n) {
        int i = dist_idx(rng);
        int j = dist_idx(rng);
        if (j == i) {
            j = (j + 1) % num_queries;
        }
        const float dist = l2Distance(ref_desc.row(i), target_desc.row(j));
        ranked.emplace_back(-dist, -1);
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    return computeAPFromLabels(ranked, positives);
}

float computeVerificationDiffSeqPaper(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const cv::Mat& negative_pool,
    int positives_target,
    int negatives_target,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries || negative_pool.rows == 0) {
        return 0.0f;
    }

    const int positives = std::min(positives_target, num_queries);
    std::vector<int> pos_indices = sampleIndices(num_queries, positives, rng);

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(positives + negatives_target));

    for (int idx : pos_indices) {
        const float dist = l2Distance(ref_desc.row(idx), target_desc.row(idx));
        ranked.emplace_back(-dist, 1);
    }

    const int max_neg = std::min(negatives_target, negative_pool.rows);
    std::uniform_int_distribution<int> dist_idx(0, negative_pool.rows - 1);
    for (int n = 0; n < max_neg; ++n) {
        int j = dist_idx(rng);
        const float dist = l2Distance(ref_desc.row(pos_indices[n % positives]), negative_pool.row(j));
        ranked.emplace_back(-dist, -1);
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    return computeAPFromLabels(ranked, positives);
}

float computeRetrieval(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    const cv::Mat& negative_pool,
    int negatives_per_query,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries) {
        return 0.0f;
    }

    int negatives = std::min(negatives_per_query, negative_pool.rows);
    auto neg_indices = sampleIndices(negative_pool.rows, negatives, rng);

    double sum_ap = 0.0;
    for (int i = 0; i < num_queries; ++i) {
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(static_cast<size_t>(num_queries + neg_indices.size()));

        for (int j = 0; j < num_queries; ++j) {
            const float dist = l2Distance(ref_desc.row(i), target_desc.row(j));
            ranked.emplace_back(dist, j == i ? 1 : 0);
        }

        for (int idx : neg_indices) {
            const float dist = l2Distance(ref_desc.row(i), negative_pool.row(idx));
            ranked.emplace_back(dist, 0);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        sum_ap += computeAPFromLabels(ranked, 1);
    }

    return static_cast<float>(sum_ap / static_cast<double>(num_queries));
}

float computeMatchingPaper(
    const cv::Mat& ref_desc,
    const cv::Mat& target_desc,
    float* out_accuracy) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_desc.rows != num_queries) {
        if (out_accuracy) *out_accuracy = 0.0f;
        return 0.0f;
    }

    cv::Mat ref_sq, tgt_sq;
    cv::multiply(ref_desc, ref_desc, ref_sq);
    cv::multiply(target_desc, target_desc, tgt_sq);
    cv::reduce(ref_sq, ref_sq, 1, cv::REDUCE_SUM, CV_32F);
    cv::reduce(tgt_sq, tgt_sq, 1, cv::REDUCE_SUM, CV_32F);

    cv::Mat dot;
    cv::gemm(ref_desc, target_desc.t(), 1.0, cv::Mat(), 0.0, dot);

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(num_queries));
    int correct = 0;
    for (int i = 0; i < num_queries; ++i) {
        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::max();
        const float ref_norm = ref_sq.at<float>(i, 0);
        for (int j = 0; j < num_queries; ++j) {
            const float dist = ref_norm + tgt_sq.at<float>(j, 0) - 2.0f * dot.at<float>(i, j);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        const int label = (best_idx == i) ? 1 : -1;
        if (label == 1) {
            correct++;
        }
        ranked.emplace_back(-best_dist, label);
    }

    if (out_accuracy) {
        *out_accuracy = static_cast<float>(correct) / static_cast<float>(num_queries);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    return computeAPFromLabels(ranked, num_queries);
}

float computeRetrievalPaper(
    const cv::Mat& ref_desc,
    const std::vector<cv::Mat>& target_descs,
    const cv::Mat& negative_pool,
    int num_queries_target,
    int num_distractors,
    std::mt19937& rng) {

    const int num_queries = ref_desc.rows;
    if (num_queries == 0 || target_descs.empty()) {
        return 0.0f;
    }

    const int queries = std::min(num_queries_target, num_queries);
    std::vector<int> query_indices = sampleIndices(num_queries, queries, rng);

    const int positives = static_cast<int>(target_descs.size());
    cv::Mat distractors = negative_pool;
    if (negative_pool.rows > num_distractors) {
        auto sampled = sampleIndices(negative_pool.rows, num_distractors, rng);
        cv::Mat subset(static_cast<int>(sampled.size()), negative_pool.cols, negative_pool.type());
        for (int r = 0; r < static_cast<int>(sampled.size()); ++r) {
            negative_pool.row(sampled[r]).copyTo(subset.row(r));
        }
        distractors = subset;
    }

    double sum_ap = 0.0;
    for (int qi : query_indices) {
        std::vector<std::pair<float, int>> ranked;
        for (const auto& target_desc : target_descs) {
            const int rows = target_desc.rows;
            for (int j = 0; j < rows; ++j) {
                const float dist = l2Distance(ref_desc.row(qi), target_desc.row(j));
                int label = 0;
                if (j == qi) {
                    label = 1;
                }
                ranked.emplace_back(-dist, label);
            }
        }
        for (int j = 0; j < distractors.rows; ++j) {
            const float dist = l2Distance(ref_desc.row(qi), distractors.row(j));
            ranked.emplace_back(-dist, -1);
        }

        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        sum_ap += computeAPWithIgnore(ranked, positives);
    }

    return static_cast<float>(sum_ap / static_cast<double>(queries));
}

void accumulateTask(TaskAccumulators& acc, float ap, const std::string& difficulty, bool is_illumination) {
    acc.overall.add(ap);
    if (difficulty == "easy") {
        acc.easy.add(ap);
        if (is_illumination) acc.illumination_easy.add(ap);
        else acc.viewpoint_easy.add(ap);
    } else if (difficulty == "hard") {
        acc.hard.add(ap);
        if (is_illumination) acc.illumination_hard.add(ap);
        else acc.viewpoint_hard.add(ap);
    } else {
        acc.tough.add(ap);
    }
    if (is_illumination) acc.illumination.add(ap);
    else acc.viewpoint.add(ap);
}

} // namespace

HPatchesBenchmark::Results HPatchesBenchmark::run(
    const Config& config,
    IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    database::DatabaseManager* database_manager,
    const ProgressCallback& progress_callback) {

    Results results;
    results.descriptor_name = extractor.name();
    results.descriptor_dimension = extractor.descriptorSize();

    auto start_time = std::chrono::high_resolution_clock::now();
    auto logStage = [&](const std::string& label, bool done = false, double seconds = 0.0) {
        if (!config.verbose) return;
        if (!done) {
            std::cout << "[PatchBenchmark] " << label << "..." << std::endl;
        } else {
            std::cout << "[PatchBenchmark] " << label << " done (" << std::fixed
                      << std::setprecision(2) << seconds << "s)" << std::endl;
        }
    };

    // Get list of scenes
    std::vector<std::string> scene_dirs;
    if (config.scenes.empty()) {
        scene_dirs = PatchLoader::listScenes(config.patches_dir);
    } else {
        scene_dirs = config.scenes;
    }

    results.num_scenes = static_cast<int>(scene_dirs.size());

    if (config.verbose) {
        std::cout << "HPatches Benchmark: " << results.descriptor_name
                  << " (" << results.descriptor_dimension << "D)" << std::endl;
        if (config.tasks.matching) {
            std::cout << "Processing " << scene_dirs.size() << " scenes (matching stage)..." << std::endl;
        } else {
            std::cout << "Loaded " << scene_dirs.size() << " scenes (matching disabled)..." << std::endl;
        }
        std::cout << "Tasks: mode=" << config.tasks.mode
                  << ", source=" << config.tasks.task_source
                  << ", split=" << config.tasks.task_split << std::endl;
    }

    // Accumulators for different categories (matching)
    Accumulator overall, easy_all, hard_all, tough_all;
    Accumulator illumination_all, viewpoint_all;
    Accumulator illumination_easy, illumination_hard;
    Accumulator viewpoint_easy, viewpoint_hard;

    TaskAccumulators verification_same;
    TaskAccumulators verification_diff;
    TaskAccumulators retrieval;

    int total_patches = 0;
    int scenes_processed = 0;

    const bool allow_parallel = (config.num_threads != 1);
    const bool run_scene_loop = (config.tasks.matching || config.tasks.mode != "paper");

    std::optional<std::chrono::high_resolution_clock::time_point> match_start;
    if (config.tasks.matching) {
        match_start = std::chrono::high_resolution_clock::now();
        logStage("Computing matching metrics");
    } else if (config.verbose) {
        logStage("Matching disabled (skipping matching stage)");
    }
    if (!run_scene_loop && config.verbose) {
        logStage("Skipping per-scene evaluation (paper tasks only)");
    }

#ifdef _OPENMP
    if (allow_parallel && config.num_threads > 0) {
        omp_set_num_threads(config.num_threads);
    }
#endif

    std::vector<std::string> illumination_scenes;
    std::vector<std::string> viewpoint_scenes;
    illumination_scenes.reserve(scene_dirs.size());
    viewpoint_scenes.reserve(scene_dirs.size());
    for (const auto& scene_dir : scene_dirs) {
        if (PatchLoader::isIlluminationScene(scene_dir)) {
            illumination_scenes.push_back(scene_dir);
        } else {
            viewpoint_scenes.push_back(scene_dir);
        }
    }

    if (run_scene_loop && allow_parallel) {
        std::atomic<int> completed{0};

        #pragma omp parallel
        {
            auto local_extractor = extractor.clone();

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < scene_dirs.size(); ++i) {
                const auto& scene_dir = scene_dirs[i];
                bool is_illumination = PatchLoader::isIlluminationScene(scene_dir);

                Accumulator local_overall, local_easy, local_hard, local_tough;
                Accumulator local_illumination_all, local_viewpoint_all;
                Accumulator local_illumination_easy, local_illumination_hard;
                Accumulator local_viewpoint_easy, local_viewpoint_hard;
                TaskAccumulators local_ver_same;
                TaskAccumulators local_ver_diff;
                TaskAccumulators local_retrieval;
                int local_total_patches = 0;

            // Evaluate each difficulty level
            std::vector<std::pair<std::string, bool>> difficulties;
            if (config.include_easy) difficulties.emplace_back("easy", true);
            if (config.include_hard) difficulties.emplace_back("hard", true);
            if (config.include_tough) difficulties.emplace_back("tough", true);

            for (const auto& [difficulty, enabled] : difficulties) {
                if (!enabled) continue;

                try {
                    auto result = evaluateScene(
                        scene_dir,
                        *local_extractor,
                        params,
                        config,
                        difficulty,
                        illumination_scenes,
                        viewpoint_scenes,
                        database_manager);

                    if (result.matching.num_patches > 0) {
                        local_total_patches += result.matching.num_patches;
                        local_overall.add(result.matching);

                        // By difficulty
                        if (difficulty == "easy") {
                            local_easy.add(result.matching);
                            if (is_illumination) local_illumination_easy.add(result.matching);
                            else local_viewpoint_easy.add(result.matching);
                        } else if (difficulty == "hard") {
                            local_hard.add(result.matching);
                            if (is_illumination) local_illumination_hard.add(result.matching);
                            else local_viewpoint_hard.add(result.matching);
                        } else {
                            local_tough.add(result.matching);
                        }

                        // By scene type
                        if (is_illumination) local_illumination_all.add(result.matching);
                        else local_viewpoint_all.add(result.matching);
                    }

                    if (config.tasks.verification && config.tasks.verification_same_seq &&
                        result.verification_same.num_patches > 0) {
                        accumulateTask(local_ver_same, result.verification_same.average_precision, difficulty, is_illumination);
                    }
                    if (config.tasks.verification && config.tasks.verification_diff_seq &&
                        result.verification_diff.num_patches > 0) {
                        accumulateTask(local_ver_diff, result.verification_diff.average_precision, difficulty, is_illumination);
                    }
                    if (config.tasks.retrieval && result.retrieval.num_patches > 0) {
                        accumulateTask(local_retrieval, result.retrieval.average_precision, difficulty, is_illumination);
                    }
                } catch (const std::exception& e) {
                    if (config.verbose) {
                        #pragma omp critical(patch_benchmark_warnings)
                        {
                            std::cerr << "Warning: Failed to evaluate " << scene_dir
                                      << " (" << difficulty << "): " << e.what() << std::endl;
                        }
                    }
                }
            }

            if (local_total_patches > 0) {
                #pragma omp critical(patch_benchmark_accumulate)
                {
                    total_patches += local_total_patches;
                    overall.sum_map += local_overall.sum_map;
                    overall.sum_accuracy += local_overall.sum_accuracy;
                    overall.count += local_overall.count;

                    easy_all.sum_map += local_easy.sum_map;
                    easy_all.sum_accuracy += local_easy.sum_accuracy;
                    easy_all.count += local_easy.count;

                    hard_all.sum_map += local_hard.sum_map;
                    hard_all.sum_accuracy += local_hard.sum_accuracy;
                    hard_all.count += local_hard.count;

                    tough_all.sum_map += local_tough.sum_map;
                    tough_all.sum_accuracy += local_tough.sum_accuracy;
                    tough_all.count += local_tough.count;

                    illumination_all.sum_map += local_illumination_all.sum_map;
                    illumination_all.sum_accuracy += local_illumination_all.sum_accuracy;
                    illumination_all.count += local_illumination_all.count;

                    viewpoint_all.sum_map += local_viewpoint_all.sum_map;
                    viewpoint_all.sum_accuracy += local_viewpoint_all.sum_accuracy;
                    viewpoint_all.count += local_viewpoint_all.count;

                    illumination_easy.sum_map += local_illumination_easy.sum_map;
                    illumination_easy.sum_accuracy += local_illumination_easy.sum_accuracy;
                    illumination_easy.count += local_illumination_easy.count;

                    illumination_hard.sum_map += local_illumination_hard.sum_map;
                    illumination_hard.sum_accuracy += local_illumination_hard.sum_accuracy;
                    illumination_hard.count += local_illumination_hard.count;

                    viewpoint_easy.sum_map += local_viewpoint_easy.sum_map;
                    viewpoint_easy.sum_accuracy += local_viewpoint_easy.sum_accuracy;
                    viewpoint_easy.count += local_viewpoint_easy.count;

                    viewpoint_hard.sum_map += local_viewpoint_hard.sum_map;
                    viewpoint_hard.sum_accuracy += local_viewpoint_hard.sum_accuracy;
                    viewpoint_hard.count += local_viewpoint_hard.count;

                    verification_same.overall.sum_map += local_ver_same.overall.sum_map;
                    verification_same.overall.count += local_ver_same.overall.count;
                    verification_same.easy.sum_map += local_ver_same.easy.sum_map;
                    verification_same.easy.count += local_ver_same.easy.count;
                    verification_same.hard.sum_map += local_ver_same.hard.sum_map;
                    verification_same.hard.count += local_ver_same.hard.count;
                    verification_same.tough.sum_map += local_ver_same.tough.sum_map;
                    verification_same.tough.count += local_ver_same.tough.count;
                    verification_same.illumination.sum_map += local_ver_same.illumination.sum_map;
                    verification_same.illumination.count += local_ver_same.illumination.count;
                    verification_same.viewpoint.sum_map += local_ver_same.viewpoint.sum_map;
                    verification_same.viewpoint.count += local_ver_same.viewpoint.count;
                    verification_same.illumination_easy.sum_map += local_ver_same.illumination_easy.sum_map;
                    verification_same.illumination_easy.count += local_ver_same.illumination_easy.count;
                    verification_same.illumination_hard.sum_map += local_ver_same.illumination_hard.sum_map;
                    verification_same.illumination_hard.count += local_ver_same.illumination_hard.count;
                    verification_same.viewpoint_easy.sum_map += local_ver_same.viewpoint_easy.sum_map;
                    verification_same.viewpoint_easy.count += local_ver_same.viewpoint_easy.count;
                    verification_same.viewpoint_hard.sum_map += local_ver_same.viewpoint_hard.sum_map;
                    verification_same.viewpoint_hard.count += local_ver_same.viewpoint_hard.count;

                    verification_diff.overall.sum_map += local_ver_diff.overall.sum_map;
                    verification_diff.overall.count += local_ver_diff.overall.count;
                    verification_diff.easy.sum_map += local_ver_diff.easy.sum_map;
                    verification_diff.easy.count += local_ver_diff.easy.count;
                    verification_diff.hard.sum_map += local_ver_diff.hard.sum_map;
                    verification_diff.hard.count += local_ver_diff.hard.count;
                    verification_diff.tough.sum_map += local_ver_diff.tough.sum_map;
                    verification_diff.tough.count += local_ver_diff.tough.count;
                    verification_diff.illumination.sum_map += local_ver_diff.illumination.sum_map;
                    verification_diff.illumination.count += local_ver_diff.illumination.count;
                    verification_diff.viewpoint.sum_map += local_ver_diff.viewpoint.sum_map;
                    verification_diff.viewpoint.count += local_ver_diff.viewpoint.count;
                    verification_diff.illumination_easy.sum_map += local_ver_diff.illumination_easy.sum_map;
                    verification_diff.illumination_easy.count += local_ver_diff.illumination_easy.count;
                    verification_diff.illumination_hard.sum_map += local_ver_diff.illumination_hard.sum_map;
                    verification_diff.illumination_hard.count += local_ver_diff.illumination_hard.count;
                    verification_diff.viewpoint_easy.sum_map += local_ver_diff.viewpoint_easy.sum_map;
                    verification_diff.viewpoint_easy.count += local_ver_diff.viewpoint_easy.count;
                    verification_diff.viewpoint_hard.sum_map += local_ver_diff.viewpoint_hard.sum_map;
                    verification_diff.viewpoint_hard.count += local_ver_diff.viewpoint_hard.count;

                    retrieval.overall.sum_map += local_retrieval.overall.sum_map;
                    retrieval.overall.count += local_retrieval.overall.count;
                    retrieval.easy.sum_map += local_retrieval.easy.sum_map;
                    retrieval.easy.count += local_retrieval.easy.count;
                    retrieval.hard.sum_map += local_retrieval.hard.sum_map;
                    retrieval.hard.count += local_retrieval.hard.count;
                    retrieval.tough.sum_map += local_retrieval.tough.sum_map;
                    retrieval.tough.count += local_retrieval.tough.count;
                    retrieval.illumination.sum_map += local_retrieval.illumination.sum_map;
                    retrieval.illumination.count += local_retrieval.illumination.count;
                    retrieval.viewpoint.sum_map += local_retrieval.viewpoint.sum_map;
                    retrieval.viewpoint.count += local_retrieval.viewpoint.count;
                    retrieval.illumination_easy.sum_map += local_retrieval.illumination_easy.sum_map;
                    retrieval.illumination_easy.count += local_retrieval.illumination_easy.count;
                    retrieval.illumination_hard.sum_map += local_retrieval.illumination_hard.sum_map;
                    retrieval.illumination_hard.count += local_retrieval.illumination_hard.count;
                    retrieval.viewpoint_easy.sum_map += local_retrieval.viewpoint_easy.sum_map;
                    retrieval.viewpoint_easy.count += local_retrieval.viewpoint_easy.count;
                    retrieval.viewpoint_hard.sum_map += local_retrieval.viewpoint_hard.sum_map;
                    retrieval.viewpoint_hard.count += local_retrieval.viewpoint_hard.count;
                }
            }

            const int current = ++completed;
            if (progress_callback) {
                #pragma omp critical(patch_benchmark_progress)
                {
                    progress_callback(current, static_cast<int>(scene_dirs.size()), scene_dir);
                }
            }
            }
        }

        scenes_processed = completed.load();
    } else if (run_scene_loop) {
        // Process each scene sequentially
        for (size_t i = 0; i < scene_dirs.size(); ++i) {
            const auto& scene_dir = scene_dirs[i];
            bool is_illumination = PatchLoader::isIlluminationScene(scene_dir);

            if (progress_callback) {
                progress_callback(static_cast<int>(i + 1), static_cast<int>(scene_dirs.size()), scene_dir);
            }

            // Evaluate each difficulty level
            std::vector<std::pair<std::string, bool>> difficulties;
            if (config.include_easy) difficulties.emplace_back("easy", true);
            if (config.include_hard) difficulties.emplace_back("hard", true);
            if (config.include_tough) difficulties.emplace_back("tough", true);

            for (const auto& [difficulty, enabled] : difficulties) {
                if (!enabled) continue;

                try {
                    auto result = evaluateScene(
                        scene_dir,
                        extractor,
                        params,
                        config,
                        difficulty,
                        illumination_scenes,
                        viewpoint_scenes,
                        database_manager);

                    if (result.matching.num_patches > 0) {
                        total_patches += result.matching.num_patches;
                        overall.add(result.matching);

                        // By difficulty
                        if (difficulty == "easy") {
                            easy_all.add(result.matching);
                            if (is_illumination) illumination_easy.add(result.matching);
                            else viewpoint_easy.add(result.matching);
                        } else if (difficulty == "hard") {
                            hard_all.add(result.matching);
                            if (is_illumination) illumination_hard.add(result.matching);
                            else viewpoint_hard.add(result.matching);
                        } else {
                            tough_all.add(result.matching);
                        }

                        // By scene type
                        if (is_illumination) illumination_all.add(result.matching);
                        else viewpoint_all.add(result.matching);
                    }

                    if (config.tasks.verification && config.tasks.verification_same_seq &&
                        result.verification_same.num_patches > 0) {
                        accumulateTask(verification_same, result.verification_same.average_precision, difficulty, is_illumination);
                    }
                    if (config.tasks.verification && config.tasks.verification_diff_seq &&
                        result.verification_diff.num_patches > 0) {
                        accumulateTask(verification_diff, result.verification_diff.average_precision, difficulty, is_illumination);
                    }
                    if (config.tasks.retrieval && result.retrieval.num_patches > 0) {
                        accumulateTask(retrieval, result.retrieval.average_precision, difficulty, is_illumination);
                    }
                } catch (const std::exception& e) {
                    if (config.verbose) {
                        std::cerr << "Warning: Failed to evaluate " << scene_dir
                                  << " (" << difficulty << "): " << e.what() << std::endl;
                    }
                }
            }

            scenes_processed++;
        }
    }

    if (match_start) {
        auto match_end = std::chrono::high_resolution_clock::now();
        logStage("Computing matching metrics", true,
                 std::chrono::duration<double>(match_end - *match_start).count());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Fill in results
    results.mAP_overall = overall.meanMAP();
    results.accuracy_overall = overall.meanAccuracy();

    results.mAP_easy = easy_all.meanMAP();
    results.mAP_hard = hard_all.meanMAP();
    results.mAP_tough = tough_all.meanMAP();

    results.mAP_illumination = illumination_all.meanMAP();
    results.mAP_viewpoint = viewpoint_all.meanMAP();

    results.mAP_illumination_easy = illumination_easy.meanMAP();
    results.mAP_illumination_hard = illumination_hard.meanMAP();
    results.mAP_viewpoint_easy = viewpoint_easy.meanMAP();
    results.mAP_viewpoint_hard = viewpoint_hard.meanMAP();

    results.verification_same_overall = verification_same.overall.meanMAP();
    results.verification_same_easy = verification_same.easy.meanMAP();
    results.verification_same_hard = verification_same.hard.meanMAP();
    results.verification_same_tough = verification_same.tough.meanMAP();
    results.verification_same_illumination = verification_same.illumination.meanMAP();
    results.verification_same_viewpoint = verification_same.viewpoint.meanMAP();
    results.verification_same_illumination_easy = verification_same.illumination_easy.meanMAP();
    results.verification_same_illumination_hard = verification_same.illumination_hard.meanMAP();
    results.verification_same_viewpoint_easy = verification_same.viewpoint_easy.meanMAP();
    results.verification_same_viewpoint_hard = verification_same.viewpoint_hard.meanMAP();

    results.verification_diff_overall = verification_diff.overall.meanMAP();
    results.verification_diff_easy = verification_diff.easy.meanMAP();
    results.verification_diff_hard = verification_diff.hard.meanMAP();
    results.verification_diff_tough = verification_diff.tough.meanMAP();
    results.verification_diff_illumination = verification_diff.illumination.meanMAP();
    results.verification_diff_viewpoint = verification_diff.viewpoint.meanMAP();
    results.verification_diff_illumination_easy = verification_diff.illumination_easy.meanMAP();
    results.verification_diff_illumination_hard = verification_diff.illumination_hard.meanMAP();
    results.verification_diff_viewpoint_easy = verification_diff.viewpoint_easy.meanMAP();
    results.verification_diff_viewpoint_hard = verification_diff.viewpoint_hard.meanMAP();

    results.retrieval_overall = retrieval.overall.meanMAP();
    results.retrieval_easy = retrieval.easy.meanMAP();
    results.retrieval_hard = retrieval.hard.meanMAP();
    results.retrieval_tough = retrieval.tough.meanMAP();
    results.retrieval_illumination = retrieval.illumination.meanMAP();
    results.retrieval_viewpoint = retrieval.viewpoint.meanMAP();
    results.retrieval_illumination_easy = retrieval.illumination_easy.meanMAP();
    results.retrieval_illumination_hard = retrieval.illumination_hard.meanMAP();
    results.retrieval_viewpoint_easy = retrieval.viewpoint_easy.meanMAP();
    results.retrieval_viewpoint_hard = retrieval.viewpoint_hard.meanMAP();

    results.num_patches = total_patches;
    results.processing_time_ms = static_cast<double>(duration.count());

    DescriptorStore descriptor_store;
    descriptor_store.db = database_manager;
    descriptor_store.descriptor_set_id = config.tasks.descriptor_cache_id;
    descriptor_store.read = config.tasks.use_cached_descriptors || config.tasks.store_descriptors_to_db;
    descriptor_store.write = config.tasks.store_descriptors_to_db;
    if (config.verbose && (descriptor_store.read || descriptor_store.write)) {
        std::cout << "[PatchBenchmark] Descriptor cache: name=" << config.tasks.descriptor_cache_name
                  << ", id=" << descriptor_store.descriptor_set_id
                  << ", read=" << (descriptor_store.read ? "true" : "false")
                  << ", write=" << (descriptor_store.write ? "true" : "false") << std::endl;
    }

    PreloadedData preloaded;
    PreloadPlan preload_plan;
    const PreloadPlan* preload_plan_ptr = nullptr;
    if (config.tasks.mode == "paper" && config.tasks.preload_descriptors) {
        auto preload_start = std::chrono::high_resolution_clock::now();
        if (config.tasks.preload_scope == "tasks") {
            logStage("Preloading descriptors for task-referenced patches");
            preload_plan = buildPreloadPlanFromTasks(config);
            preload_plan_ptr = &preload_plan;
            if (config.verbose) {
                std::cout << "[PatchBenchmark] Preload scope: tasks ("
                          << preload_plan.total_targets << " target stacks)" << std::endl;
            }
        } else {
            logStage("Preloading descriptors for all scenes");
            if (config.verbose) {
                const size_t all_targets = scene_dirs.size() * 6 * 3;
                std::cout << "[PatchBenchmark] Preload scope: all (approx "
                          << all_targets << " target stacks)" << std::endl;
            }
        }
        preloaded = preloadDescriptors(scene_dirs, config, extractor, params, preload_plan_ptr, &descriptor_store);
        auto preload_end = std::chrono::high_resolution_clock::now();
        logStage(config.tasks.preload_scope == "tasks"
                     ? "Preloading descriptors for task-referenced patches"
                     : "Preloading descriptors for all scenes",
                 true,
                 std::chrono::duration<double>(preload_end - preload_start).count());
    }

    if (config.tasks.mode == "paper") {
        const bool use_tasks = (config.tasks.task_source != "random" &&
                                !config.tasks.verification_pos_pairs.empty());
        auto mean = [](float a, float b, float c) {
            return (a + b + c) / 3.0f;
        };

        if (config.tasks.verification) {
            auto ver_start = std::chrono::high_resolution_clock::now();
            logStage("Computing verification metrics (paper mode)");
            if (config.tasks.verification_same_seq) {
                results.verification_same_easy = config.include_easy
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_intra_pairs,
                              "easy",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "easy", config, extractor, params, false,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_same_hard = config.include_hard
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_intra_pairs,
                              "hard",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "hard", config, extractor, params, false,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_same_tough = config.include_tough
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_intra_pairs,
                              "tough",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "tough", config, extractor, params, false,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_same_overall = mean(
                    results.verification_same_easy,
                    results.verification_same_hard,
                    results.verification_same_tough);
            }
            if (config.tasks.verification_diff_seq) {
                results.verification_diff_easy = config.include_easy
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_inter_pairs,
                              "easy",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "easy", config, extractor, params, true,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_diff_hard = config.include_hard
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_inter_pairs,
                              "hard",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "hard", config, extractor, params, true,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_diff_tough = config.include_tough
                    ? (use_tasks
                        ? computeVerificationFromTasks(
                              config.tasks.verification_pos_pairs,
                              config.tasks.verification_neg_inter_pairs,
                              "tough",
                              "full",
                              config,
                              extractor,
                              params,
                              config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                        : computeVerificationGlobal(scene_dirs, "tough", config, extractor, params, true,
                                                    config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                    : 0.0f;
                results.verification_diff_overall = mean(
                    results.verification_diff_easy,
                    results.verification_diff_hard,
                    results.verification_diff_tough);
            }

            results.verification_same_illumination = mean(
                config.include_easy ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "easy", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "easy", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "hard", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "hard", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "tough", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "tough", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f);
            results.verification_same_viewpoint = mean(
                config.include_easy ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "easy", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "easy", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "hard", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "hard", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_intra_pairs,
                                                   "tough", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "tough", config, extractor, params, false,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f);
            results.verification_diff_illumination = mean(
                config.include_easy ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "easy", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "easy", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "hard", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "hard", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "tough", "illum", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(illumination_scenes, "tough", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f);
            results.verification_diff_viewpoint = mean(
                config.include_easy ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "easy", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "easy", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "hard", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "hard", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeVerificationFromTasks(config.tasks.verification_pos_pairs,
                                                   config.tasks.verification_neg_inter_pairs,
                                                   "tough", "view", config, extractor, params,
                                                   config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeVerificationGlobal(viewpoint_scenes, "tough", config, extractor, params, true,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f);
            auto ver_end = std::chrono::high_resolution_clock::now();
            logStage("Computing verification metrics (paper mode)", true,
                     std::chrono::duration<double>(ver_end - ver_start).count());
        }

        if (config.tasks.retrieval) {
            auto ret_start = std::chrono::high_resolution_clock::now();
            logStage("Computing retrieval metrics (paper mode)");
            results.retrieval_easy = config.include_easy
                ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "easy", "full", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(scene_dirs, "easy", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                : 0.0f;
            results.retrieval_hard = config.include_hard
                ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "hard", "full", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(scene_dirs, "hard", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                : 0.0f;
            results.retrieval_tough = config.include_tough
                ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "tough", "full", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(scene_dirs, "tough", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "all"))
                : 0.0f;
            results.retrieval_overall = mean(
                results.retrieval_easy,
                results.retrieval_hard,
                results.retrieval_tough);

            results.retrieval_illumination = mean(
                config.include_easy ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "easy", "illum", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(illumination_scenes, "easy", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "hard", "illum", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(illumination_scenes, "hard", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "tough", "illum", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(illumination_scenes, "tough", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "illum")) : 0.0f);
            results.retrieval_viewpoint = mean(
                config.include_easy ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "easy", "view", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(viewpoint_scenes, "easy", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_hard ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "hard", "view", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(viewpoint_scenes, "hard", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f,
                config.include_tough ? (use_tasks
                    ? computeRetrievalFromTasks(config.tasks.retrieval_queries,
                                                config.tasks.retrieval_distractors,
                                                "tough", "view", config, extractor, params,
                                                config.tasks.preload_descriptors ? &preloaded : nullptr, &descriptor_store)
                    : computeRetrievalGlobal(viewpoint_scenes, "tough", config, extractor, params,
                                             config.tasks.preload_descriptors ? &preloaded : nullptr, "view")) : 0.0f);
            auto ret_end = std::chrono::high_resolution_clock::now();
            logStage("Computing retrieval metrics (paper mode)", true,
                     std::chrono::duration<double>(ret_end - ret_start).count());
        }
    }

    if (config.print_results) {
        printResults(results);
    }

    return results;
}

HPatchesBenchmark::SceneResults HPatchesBenchmark::evaluateScene(
    const std::string& scene_dir,
    IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    const Config& config,
    const std::string& difficulty,
    const std::vector<std::string>& illumination_scenes,
    const std::vector<std::string>& viewpoint_scenes,
    database::DatabaseManager* database_manager) {

    SceneResults scene_results;

    // Load the scene
    auto scene = PatchLoader::loadScene(scene_dir, config.color);
    const bool is_illumination = PatchLoader::isIlluminationScene(scene_dir);
    const std::string scene_label = std::filesystem::path(scene_dir).filename().string();

    // Get the appropriate patch sets based on difficulty
    const std::map<std::string, PatchLoader::PatchSet>* target_sets = nullptr;
    if (difficulty == "easy") {
        target_sets = &scene.easy;
    } else if (difficulty == "hard") {
        target_sets = &scene.hard;
    } else if (difficulty == "tough") {
        target_sets = &scene.tough;
    } else {
        throw std::invalid_argument("Unknown difficulty: " + difficulty);
    }

    if (target_sets->empty()) {
        return scene_results;  // No patches for this difficulty
    }

    DescriptorStore store;
    store.db = database_manager;
    store.read = config.tasks.use_cached_descriptors;
    store.write = config.tasks.store_descriptors_to_db;
    store.descriptor_set_id = config.tasks.descriptor_cache_id;

    DescriptorCache cache(12);
    std::unordered_map<std::string, cv::Mat> ref_cache;

    auto loadFromStore = [&](const std::string& target_key) -> cv::Mat {
        if (!store.enabled() || !store.read) {
            return {};
        }
        const std::string key = descriptorKey(scene_dir, difficulty, target_key);
        std::lock_guard<std::mutex> lock(store.mutex);
        if (const auto memo_it = store.memo.find(key); memo_it != store.memo.end()) {
            return memo_it->second;
        }
        auto db_mat = store.db->loadPatchBenchmarkDescriptor(
            store.descriptor_set_id, scene_label, difficulty, target_key);
        if (db_mat) {
            store.memo.emplace(key, *db_mat);
            return *db_mat;
        }
        return {};
    };

    auto storeToDb = [&](const std::string& target_key, const cv::Mat& desc) {
        if (!store.enabled() || !store.write || desc.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(store.mutex);
        store.db->storePatchBenchmarkDescriptor(
            store.descriptor_set_id, scene_label, difficulty, target_key, desc);
        store.memo.emplace(descriptorKey(scene_dir, difficulty, target_key), desc);
    };

    cv::Mat ref_descriptors = loadFromStore("ref");
    if (ref_descriptors.empty()) {
        ref_descriptors = extractor.extractFromPatches(scene.ref.patches, params);
        storeToDb("ref", ref_descriptors);
    }

    // Evaluate against each target set (e1-e5, h1-h5, or t1-t5)
    PatchMetrics::MatchResult matching_combined;
    matching_combined.num_patches = 0;
    double sum_ap = 0.0;
    double sum_accuracy = 0.0;
    int count = 0;

    double ver_same_sum = 0.0;
    int ver_same_count = 0;
    double ver_diff_sum = 0.0;
    int ver_diff_count = 0;
    double retrieval_sum = 0.0;
    int retrieval_count = 0;

    const std::string mode = config.tasks.mode;

    for (const auto& [key, patch_set] : *target_sets) {
        if (patch_set.patches.size() != scene.ref.patches.size()) {
            continue;  // Skip mismatched sets
        }

        cv::Mat target_descriptors = loadFromStore(key);
        if (target_descriptors.empty()) {
            target_descriptors = extractor.extractFromPatches(patch_set.patches, params);
            storeToDb(key, target_descriptors);
        }

        if (config.tasks.matching) {
            if (mode == "paper") {
                float accuracy = 0.0f;
                const float ap = computeMatchingPaper(ref_descriptors, target_descriptors, &accuracy);
                matching_combined.num_patches += ref_descriptors.rows;
                matching_combined.correct_matches += static_cast<int>(accuracy * ref_descriptors.rows);
                sum_ap += ap;
                sum_accuracy += accuracy;
                count++;
            } else {
                auto result = PatchMetrics::computeMatching(ref_descriptors, target_descriptors);
                matching_combined.num_patches += result.num_patches;
                matching_combined.correct_matches += result.correct_matches;
                sum_ap += result.average_precision;
                sum_accuracy += result.match_accuracy;
                count++;
            }
        }

        std::mt19937 rng(config.tasks.random_seed +
                         static_cast<unsigned int>(std::hash<std::string>{}(scene_dir + difficulty + key + mode)));

        if (config.tasks.verification && config.tasks.verification_same_seq) {
            float ap = 0.0f;
            if (mode == "paper") {
                ap = computeVerificationSameSeqPaper(
                    ref_descriptors,
                    target_descriptors,
                    config.tasks.verification_num_positives,
                    config.tasks.verification_num_negatives,
                    rng);
            } else {
                ap = computeVerificationSameSeq(
                    ref_descriptors, target_descriptors, config.tasks.verification_negatives_per_query, rng);
            }
            ver_same_sum += ap;
            ver_same_count++;
        }

        if ((config.tasks.verification && config.tasks.verification_diff_seq) || config.tasks.retrieval) {
            cv::Mat diff_pool;
            if (config.tasks.verification && config.tasks.verification_diff_seq) {
                diff_pool = buildDiffSeqPool(
                    scene_dir,
                    difficulty,
                    key,
                    mode == "paper" ? config.tasks.verification_num_negatives : config.tasks.verification_negatives_per_query,
                    is_illumination,
                    illumination_scenes,
                    viewpoint_scenes,
                    config,
                    extractor,
                    params,
                    static_cast<unsigned int>(config.tasks.random_seed +
                                              std::hash<std::string>{}(scene_dir + difficulty + key + "ver")));
                if (!diff_pool.empty()) {
                    float ap = 0.0f;
                    if (mode == "paper") {
                        ap = computeVerificationDiffSeqPaper(
                            ref_descriptors,
                            target_descriptors,
                            diff_pool,
                            config.tasks.verification_num_positives,
                            config.tasks.verification_num_negatives,
                            rng);
                    } else {
                        ap = computeVerificationDiffSeq(
                            ref_descriptors, target_descriptors, diff_pool,
                            config.tasks.verification_negatives_per_query, rng);
                    }
                    ver_diff_sum += ap;
                    ver_diff_count++;
                }
            }

            if (config.tasks.retrieval) {
                if (diff_pool.empty() ||
                    diff_pool.rows < (mode == "paper" ? config.tasks.retrieval_num_distractors
                                                      : config.tasks.retrieval_negatives_per_query)) {
                    diff_pool = buildDiffSeqPool(
                        scene_dir,
                        difficulty,
                        key,
                        mode == "paper" ? config.tasks.retrieval_num_distractors
                                        : config.tasks.retrieval_negatives_per_query,
                        is_illumination,
                        illumination_scenes,
                        viewpoint_scenes,
                        config,
                        extractor,
                        params,
                        static_cast<unsigned int>(config.tasks.random_seed +
                                                  std::hash<std::string>{}(scene_dir + difficulty + key + "ret")));
                }
                if (!diff_pool.empty()) {
                    float ap = 0.0f;
                    if (mode == "paper") {
                        std::vector<cv::Mat> target_descs;
                        target_descs.reserve(target_sets->size());
                        for (const auto& [key2, patch_set2] : *target_sets) {
                            if (patch_set2.patches.size() != scene.ref.patches.size()) {
                                continue;
                            }
                            target_descs.push_back(extractor.extractFromPatches(patch_set2.patches, params));
                        }
                        ap = computeRetrievalPaper(
                            ref_descriptors,
                            target_descs,
                            diff_pool,
                            config.tasks.retrieval_num_queries,
                            config.tasks.retrieval_num_distractors,
                            rng);
                    } else {
                        ap = computeRetrieval(
                            ref_descriptors, target_descriptors, diff_pool,
                            config.tasks.retrieval_negatives_per_query, rng);
                    }
                    retrieval_sum += ap;
                    retrieval_count++;
                }
            }
        }
    }

    if (count > 0) {
        matching_combined.average_precision = static_cast<float>(sum_ap / count);
        matching_combined.match_accuracy = static_cast<float>(sum_accuracy / count);
    }

    scene_results.matching = matching_combined;
    if (ver_same_count > 0) {
        scene_results.verification_same.num_patches = ref_descriptors.rows;
        scene_results.verification_same.average_precision = static_cast<float>(ver_same_sum / ver_same_count);
    }
    if (ver_diff_count > 0) {
        scene_results.verification_diff.num_patches = ref_descriptors.rows;
        scene_results.verification_diff.average_precision = static_cast<float>(ver_diff_sum / ver_diff_count);
    }
    if (retrieval_count > 0) {
        scene_results.retrieval.num_patches = ref_descriptors.rows;
        scene_results.retrieval.average_precision = static_cast<float>(retrieval_sum / retrieval_count);
    }

    return scene_results;
}

void HPatchesBenchmark::printResults(const Results& results) {
    std::cout << formatResults(results);
}

std::string HPatchesBenchmark::formatResults(const Results& results) {
    std::ostringstream oss;

    oss << "\n";
    oss << "========================================\n";
    oss << "HPatches Patch Benchmark Results\n";
    oss << "========================================\n";
    oss << "Descriptor: " << results.descriptor_name
        << " (" << results.descriptor_dimension << "D)\n";
    oss << "Scenes: " << results.num_scenes
        << ", Patches: " << results.num_patches << "\n";
    oss << "Time: " << std::fixed << std::setprecision(1)
        << results.processing_time_ms / 1000.0 << "s\n";
    oss << "----------------------------------------\n";

    auto pct = [](float v) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << (v * 100.0f) << "%";
        return s.str();
    };

    oss << "Overall mAP:       " << pct(results.mAP_overall) << "\n";
    oss << "Overall Accuracy:  " << pct(results.accuracy_overall) << "\n";
    oss << "----------------------------------------\n";
    oss << "By Difficulty:\n";
    oss << "  Easy:   " << pct(results.mAP_easy) << "\n";
    oss << "  Hard:   " << pct(results.mAP_hard) << "\n";
    if (results.mAP_tough > 0) {
        oss << "  Tough:  " << pct(results.mAP_tough) << "\n";
    }
    oss << "----------------------------------------\n";
    oss << "By Scene Type:\n";
    oss << "  Illumination: " << pct(results.mAP_illumination) << "\n";
    oss << "  Viewpoint:    " << pct(results.mAP_viewpoint) << "\n";
    oss << "----------------------------------------\n";
    oss << "Detailed Breakdown:\n";
    oss << "  Illumination Easy: " << pct(results.mAP_illumination_easy) << "\n";
    oss << "  Illumination Hard: " << pct(results.mAP_illumination_hard) << "\n";
    oss << "  Viewpoint Easy:    " << pct(results.mAP_viewpoint_easy) << "\n";
    oss << "  Viewpoint Hard:    " << pct(results.mAP_viewpoint_hard) << "\n";
    oss << "----------------------------------------\n";
    oss << "Verification (SAMESEQ) mAP: " << pct(results.verification_same_overall) << "\n";
    oss << "  Easy:   " << pct(results.verification_same_easy) << "\n";
    oss << "  Hard:   " << pct(results.verification_same_hard) << "\n";
    if (results.verification_same_tough > 0) {
        oss << "  Tough:  " << pct(results.verification_same_tough) << "\n";
    }
    oss << "  Illumination: " << pct(results.verification_same_illumination) << "\n";
    oss << "  Viewpoint:    " << pct(results.verification_same_viewpoint) << "\n";
    oss << "----------------------------------------\n";
    oss << "Verification (DIFFSEQ) mAP: " << pct(results.verification_diff_overall) << "\n";
    oss << "  Easy:   " << pct(results.verification_diff_easy) << "\n";
    oss << "  Hard:   " << pct(results.verification_diff_hard) << "\n";
    if (results.verification_diff_tough > 0) {
        oss << "  Tough:  " << pct(results.verification_diff_tough) << "\n";
    }
    oss << "  Illumination: " << pct(results.verification_diff_illumination) << "\n";
    oss << "  Viewpoint:    " << pct(results.verification_diff_viewpoint) << "\n";
    oss << "----------------------------------------\n";
    oss << "Retrieval (DIFFSEQ) mAP: " << pct(results.retrieval_overall) << "\n";
    oss << "  Easy:   " << pct(results.retrieval_easy) << "\n";
    oss << "  Hard:   " << pct(results.retrieval_hard) << "\n";
    if (results.retrieval_tough > 0) {
        oss << "  Tough:  " << pct(results.retrieval_tough) << "\n";
    }
    oss << "  Illumination: " << pct(results.retrieval_illumination) << "\n";
    oss << "  Viewpoint:    " << pct(results.retrieval_viewpoint) << "\n";
    oss << "========================================\n";

    return oss.str();
}

} // namespace thesis_project::benchmark
