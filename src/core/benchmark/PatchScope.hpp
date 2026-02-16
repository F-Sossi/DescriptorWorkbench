#pragma once

#include "BenchmarkTypes.hpp"
#include <string>
#include <unordered_set>
#include <vector>
#include <functional>

namespace thesis_project::benchmark {

/**
 * @brief Key identifying a specific patch set (scene + difficulty + target)
 *
 * For HPatches, patches are organized as:
 * - scene: "i_ajuntament", "v_there", etc.
 * - difficulty: "easy", "hard", "tough"
 * - target: "ref", "e1", "e2", ..., "e5", "h1", ..., "t5"
 *
 * Note: In task files, targets use indices 0-5:
 * - 0 = ref (for any difficulty)
 * - 1-5 = e1-e5 (easy), h1-h5 (hard), or t1-t5 (tough)
 */
struct PatchKey {
    std::string scene;      ///< Scene name (e.g., "i_ajuntament")
    std::string difficulty; ///< "easy", "hard", or "tough"
    std::string target;     ///< "ref", "e1", "h3", "t5", etc.

    bool operator==(const PatchKey& other) const {
        return scene == other.scene &&
               difficulty == other.difficulty &&
               target == other.target;
    }

    bool operator!=(const PatchKey& other) const {
        return !(*this == other);
    }

    /// Convert to string representation: "scene|difficulty|target"
    std::string toString() const {
        return scene + "|" + difficulty + "|" + target;
    }

    /// Parse from string representation
    static PatchKey fromString(const std::string& s);

    /// Convert task file target index (0-5) to target name for given difficulty
    static std::string targetFromIndex(int idx, const std::string& difficulty);

    /// Get the difficulty prefix for target names ("e", "h", "t")
    static std::string difficultyPrefix(const std::string& difficulty);
};

/// Hash function for PatchKey (for use in unordered containers)
struct PatchKeyHash {
    size_t operator()(const PatchKey& k) const {
        size_t h1 = std::hash<std::string>{}(k.scene);
        size_t h2 = std::hash<std::string>{}(k.difficulty);
        size_t h3 = std::hash<std::string>{}(k.target);
        // Combine hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

/**
 * @brief Determines which patch sets need to be loaded/extracted
 *
 * PatchScope computes the minimal set of patches needed based on:
 * - Enabled tasks (matching, verification, retrieval)
 * - Enabled difficulties (easy, hard, tough)
 * - Task file contents (for verification/retrieval scoping)
 *
 * For matching: all patches in all scenes are needed
 * For verification/retrieval only: only patches referenced by task files
 */
class PatchScope {
public:
    using KeySet = std::unordered_set<PatchKey, PatchKeyHash>;

    /// Default constructor - empty scope
    PatchScope() = default;

    /// Build scope based on config and available scenes
    static PatchScope build(
        const Config& config,
        const std::vector<std::string>& scene_dirs);

    /// Build scope for all patches in given scenes (matching task)
    static PatchScope allPatches(
        const std::vector<std::string>& scene_dirs,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Build scope from verification task pairs
    static PatchScope fromVerificationTasks(
        const std::vector<VerificationTaskPair>& pos_pairs,
        const std::vector<VerificationTaskPair>& neg_intra_pairs,
        const std::vector<VerificationTaskPair>& neg_inter_pairs,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Build scope from retrieval task items
    static PatchScope fromRetrievalTasks(
        const std::vector<RetrievalTaskItem>& queries,
        const std::vector<RetrievalTaskItem>& distractors,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Add a single key to the scope
    void add(const PatchKey& key) { keys_.insert(key); }

    /// Add all keys from another scope (union)
    void merge(const PatchScope& other);

    /// Check if scope contains a key
    bool contains(const PatchKey& key) const {
        return keys_.find(key) != keys_.end();
    }

    /// Check if scope contains patches for a scene
    bool containsScene(const std::string& scene) const;

    /// Get number of patch sets in scope
    size_t size() const { return keys_.size(); }

    /// Check if scope is empty
    bool empty() const { return keys_.empty(); }

    /// Get all unique scene names
    std::vector<std::string> scenes() const;

    /// Get all keys for a specific scene
    std::vector<PatchKey> keysForScene(const std::string& scene) const;

    /// Get all difficulties used in scope
    std::vector<std::string> difficulties() const;

    /// Iteration support
    KeySet::const_iterator begin() const { return keys_.begin(); }
    KeySet::const_iterator end() const { return keys_.end(); }

    /// Get underlying key set (for advanced use)
    const KeySet& keys() const { return keys_; }

private:
    KeySet keys_;

    /// Add all patches for a scene directory
    void addAllForScene(
        const std::string& scene_dir,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Add patches from a verification pair
    void addFromVerificationPair(
        const VerificationTaskPair& pair,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Add patches from a retrieval item (query or distractor)
    void addFromRetrievalItem(
        const RetrievalTaskItem& item,
        bool include_easy,
        bool include_hard,
        bool include_tough);

    /// Extract scene name from scene directory path
    static std::string sceneNameFromPath(const std::string& scene_dir);
};

} // namespace thesis_project::benchmark
