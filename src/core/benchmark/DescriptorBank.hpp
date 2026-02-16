#pragma once

#include "PatchScope.hpp"
#include "BenchmarkTypes.hpp"
#include "core/patches/PatchDescriptorExtractor.hpp"
#include "core/patches/PatchLoader.hpp"
#include <thesis_project/types.hpp>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <functional>
#include <mutex>

namespace thesis_project {
namespace database { class DatabaseManager; }
namespace benchmark {

/**
 * @brief Centralized storage for patch descriptors
 *
 * DescriptorBank manages descriptor extraction, caching, and access:
 * - Loads descriptors from database cache if available
 * - Extracts missing descriptors using provided extractor
 * - Stores descriptors to database atomically (all or nothing)
 * - Provides thread-safe access to descriptors by PatchKey
 *
 * Usage:
 * @code
 * PatchScope scope = PatchScope::build(config, scene_dirs);
 * DescriptorBank bank(scope);
 *
 * // Optional: load from cache
 * if (config.use_cached_descriptors) {
 *     bank.loadFromDatabase(db, cache_id);
 * }
 *
 * // Extract any missing descriptors
 * bank.extractMissing(patches_dir, color, extractor, params);
 *
 * // Optional: store to cache
 * if (config.store_descriptors_to_db) {
 *     bank.storeToDatabase(db, cache_id);
 * }
 *
 * // Access descriptors
 * const cv::Mat& desc = bank.get({"i_ajuntament", "easy", "ref"});
 * @endcode
 */
class DescriptorBank {
public:
    using ProgressCallback = std::function<void(int current, int total, const std::string& message)>;

    /**
     * @brief Construct a DescriptorBank for the given scope
     * @param scope Set of PatchKeys that will be needed
     */
    explicit DescriptorBank(const PatchScope& scope);

    /**
     * @brief Load descriptors from database cache
     *
     * Loads any descriptors that exist in the database for the given cache.
     * Missing descriptors are marked for later extraction.
     *
     * @param db Database manager
     * @param cache_id Descriptor set ID from database
     * @return Number of descriptors loaded
     */
    size_t loadFromDatabase(database::DatabaseManager& db, int cache_id);

    /**
     * @brief Extract all missing descriptors
     *
     * Iterates over patches in the scope that haven't been loaded,
     * extracts descriptors, and stores them in the bank.
     *
     * @param patches_dir Path to HPatches patches directory
     * @param color Whether to load patches as color (true) or grayscale (false)
     * @param extractor Descriptor extractor to use
     * @param params Descriptor parameters
     * @param progress Optional progress callback
     */
    void extractMissing(
        const std::string& patches_dir,
        bool color,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params,
        const ProgressCallback& progress = nullptr);

    /**
     * @brief Store all descriptors to database
     *
     * Stores all loaded/extracted descriptors to the database cache.
     * Should only be called after all extraction is complete.
     *
     * @param db Database manager
     * @param cache_id Descriptor set ID to store under
     * @return true if all descriptors were stored successfully
     */
    bool storeToDatabase(database::DatabaseManager& db, int cache_id);

    /**
     * @brief Get descriptor matrix for a patch set
     *
     * Returns the full NxD descriptor matrix for the given patch set.
     * Returns empty matrix if not loaded/extracted.
     *
     * @param key PatchKey identifying the patch set
     * @return Reference to descriptor matrix (empty if not found)
     */
    const cv::Mat& get(const PatchKey& key) const;

    /**
     * @brief Get descriptor matrix by components
     * @param scene Scene name
     * @param difficulty "easy", "hard", or "tough"
     * @param target "ref", "e1", "h3", etc.
     * @return Reference to descriptor matrix (empty if not found)
     */
    const cv::Mat& get(const std::string& scene,
                       const std::string& difficulty,
                       const std::string& target) const;

    /**
     * @brief Get a single descriptor row
     * @param key PatchKey identifying the patch set
     * @param idx Row index within the descriptor matrix
     * @return Copy of the descriptor row (empty if not found or out of range)
     */
    cv::Mat getRow(const PatchKey& key, int idx) const;

    /**
     * @brief Check if all scope entries have been loaded/extracted
     * @return true if no missing entries
     */
    bool isComplete() const { return missing_.empty(); }

    /**
     * @brief Get count of loaded/extracted descriptors
     */
    size_t loadedCount() const { return descriptors_.size(); }

    /**
     * @brief Get count of missing descriptors
     */
    size_t missingCount() const { return missing_.size(); }

    /**
     * @brief Get the scope this bank was created for
     */
    const PatchScope& scope() const { return scope_; }

    /**
     * @brief Get descriptor dimension (from first loaded descriptor)
     * @return Descriptor dimension, or 0 if no descriptors loaded
     */
    int descriptorDimension() const;

    /**
     * @brief Get total number of patch rows across all descriptors
     */
    int totalPatches() const;

private:
    PatchScope scope_;
    std::unordered_map<PatchKey, cv::Mat, PatchKeyHash> descriptors_;
    std::unordered_set<PatchKey, PatchKeyHash> missing_;
    cv::Mat empty_mat_;  // Returned for missing keys
    mutable std::mutex mutex_;

    /// Group scope keys by scene for efficient loading
    std::unordered_map<std::string, std::vector<PatchKey>> groupByScene() const;

    /// Load patches and extract descriptors for a scene
    void extractForScene(
        const std::string& scene_dir,
        bool color,
        patches::IPatchDescriptorExtractor& extractor,
        const DescriptorParams& params);
};

} // namespace benchmark
} // namespace thesis_project
