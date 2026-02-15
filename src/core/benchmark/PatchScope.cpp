#include "PatchScope.hpp"
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace thesis_project::benchmark {

// -----------------------------------------------------------------------------
// PatchKey implementation
// -----------------------------------------------------------------------------

PatchKey PatchKey::fromString(const std::string& s) {
    PatchKey key;
    std::istringstream iss(s);
    std::string token;

    if (std::getline(iss, token, '|')) key.scene = token;
    if (std::getline(iss, token, '|')) key.difficulty = token;
    if (std::getline(iss, token, '|')) key.target = token;

    return key;
}

std::string PatchKey::targetFromIndex(int idx, const std::string& difficulty) {
    if (idx == 0) {
        return "ref";
    }

    std::string prefix = difficultyPrefix(difficulty);
    return prefix + std::to_string(idx);
}

std::string PatchKey::difficultyPrefix(const std::string& difficulty) {
    if (difficulty == "easy") return "e";
    if (difficulty == "hard") return "h";
    if (difficulty == "tough") return "t";
    throw std::invalid_argument("Unknown difficulty: " + difficulty);
}

// -----------------------------------------------------------------------------
// PatchScope implementation
// -----------------------------------------------------------------------------

std::string PatchScope::sceneNameFromPath(const std::string& scene_dir) {
    std::filesystem::path p(scene_dir);
    return p.filename().string();
}

PatchScope PatchScope::build(
    const Config& config,
    const std::vector<std::string>& scene_dirs) {

    PatchScope scope;

    // If matching is enabled, we need ALL patches
    if (config.matching_enabled) {
        scope = allPatches(
            scene_dirs,
            config.include_easy,
            config.include_hard,
            config.include_tough);
    }

    // Add patches from verification tasks (if verification-only or need more scope)
    if (config.verification_enabled && !config.verification_pos_pairs.empty()) {
        auto verif_scope = fromVerificationTasks(
            config.verification_pos_pairs,
            config.verification_neg_intra_pairs,
            config.verification_neg_inter_pairs,
            config.include_easy,
            config.include_hard,
            config.include_tough);
        scope.merge(verif_scope);
    }

    // Add patches from retrieval tasks
    if (config.retrieval_enabled && !config.retrieval_queries.empty()) {
        auto retr_scope = fromRetrievalTasks(
            config.retrieval_queries,
            config.retrieval_distractors,
            config.include_easy,
            config.include_hard,
            config.include_tough);
        scope.merge(retr_scope);
    }

    return scope;
}

PatchScope PatchScope::allPatches(
    const std::vector<std::string>& scene_dirs,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    PatchScope scope;
    for (const auto& scene_dir : scene_dirs) {
        scope.addAllForScene(scene_dir, include_easy, include_hard, include_tough);
    }
    return scope;
}

PatchScope PatchScope::fromVerificationTasks(
    const std::vector<VerificationTaskPair>& pos_pairs,
    const std::vector<VerificationTaskPair>& neg_intra_pairs,
    const std::vector<VerificationTaskPair>& neg_inter_pairs,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    PatchScope scope;

    for (const auto& pair : pos_pairs) {
        scope.addFromVerificationPair(pair, include_easy, include_hard, include_tough);
    }
    for (const auto& pair : neg_intra_pairs) {
        scope.addFromVerificationPair(pair, include_easy, include_hard, include_tough);
    }
    for (const auto& pair : neg_inter_pairs) {
        scope.addFromVerificationPair(pair, include_easy, include_hard, include_tough);
    }

    return scope;
}

PatchScope PatchScope::fromRetrievalTasks(
    const std::vector<RetrievalTaskItem>& queries,
    const std::vector<RetrievalTaskItem>& distractors,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    PatchScope scope;

    for (const auto& item : queries) {
        scope.addFromRetrievalItem(item, include_easy, include_hard, include_tough);
    }
    for (const auto& item : distractors) {
        scope.addFromRetrievalItem(item, include_easy, include_hard, include_tough);
    }

    return scope;
}

void PatchScope::merge(const PatchScope& other) {
    for (const auto& key : other.keys_) {
        keys_.insert(key);
    }
}

bool PatchScope::containsScene(const std::string& scene) const {
    for (const auto& key : keys_) {
        if (key.scene == scene) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> PatchScope::scenes() const {
    std::unordered_set<std::string> unique_scenes;
    for (const auto& key : keys_) {
        unique_scenes.insert(key.scene);
    }

    std::vector<std::string> result(unique_scenes.begin(), unique_scenes.end());
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<PatchKey> PatchScope::keysForScene(const std::string& scene) const {
    std::vector<PatchKey> result;
    for (const auto& key : keys_) {
        if (key.scene == scene) {
            result.push_back(key);
        }
    }
    return result;
}

std::vector<std::string> PatchScope::difficulties() const {
    std::unordered_set<std::string> unique_diffs;
    for (const auto& key : keys_) {
        unique_diffs.insert(key.difficulty);
    }

    std::vector<std::string> result(unique_diffs.begin(), unique_diffs.end());
    std::sort(result.begin(), result.end());
    return result;
}

void PatchScope::addAllForScene(
    const std::string& scene_dir,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    std::string scene = sceneNameFromPath(scene_dir);

    // Always add ref for each enabled difficulty
    // (ref is shared across difficulties but we track per-difficulty for consistency)

    if (include_easy) {
        // Add ref and e1-e5
        keys_.insert({scene, "easy", "ref"});
        for (int i = 1; i <= 5; ++i) {
            keys_.insert({scene, "easy", "e" + std::to_string(i)});
        }
    }

    if (include_hard) {
        // Add ref and h1-h5
        keys_.insert({scene, "hard", "ref"});
        for (int i = 1; i <= 5; ++i) {
            keys_.insert({scene, "hard", "h" + std::to_string(i)});
        }
    }

    if (include_tough) {
        // Add ref and t1-t5
        keys_.insert({scene, "tough", "ref"});
        for (int i = 1; i <= 5; ++i) {
            keys_.insert({scene, "tough", "t" + std::to_string(i)});
        }
    }
}

void PatchScope::addFromVerificationPair(
    const VerificationTaskPair& pair,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    // For each difficulty level, add the referenced patches
    // The task pair uses target indices 0-5, we convert to actual target names

    auto addForDifficulty = [&](const std::string& difficulty) {
        // First patch
        std::string target1 = PatchKey::targetFromIndex(pair.t1, difficulty);
        keys_.insert({pair.s1, difficulty, target1});

        // Second patch
        std::string target2 = PatchKey::targetFromIndex(pair.t2, difficulty);
        keys_.insert({pair.s2, difficulty, target2});
    };

    if (include_easy) addForDifficulty("easy");
    if (include_hard) addForDifficulty("hard");
    if (include_tough) addForDifficulty("tough");
}

void PatchScope::addFromRetrievalItem(
    const RetrievalTaskItem& item,
    bool include_easy,
    bool include_hard,
    bool include_tough) {

    // Retrieval queries/distractors always reference "ref" patches
    // But we need to add for each difficulty since the same ref is used
    // across all difficulty evaluations

    if (include_easy) {
        keys_.insert({item.s, "easy", "ref"});
    }
    if (include_hard) {
        keys_.insert({item.s, "hard", "ref"});
    }
    if (include_tough) {
        keys_.insert({item.s, "tough", "ref"});
    }
}

} // namespace thesis_project::benchmark
