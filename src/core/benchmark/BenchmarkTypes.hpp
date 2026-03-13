#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace thesis_project::benchmark {

/**
 * @brief Verification task pair from HPatches task files
 *
 * Represents a pair of patches for verification:
 * - Positive pairs: same patch idx across different targets (idx1 == idx2)
 * - Negative intra: different patches from same sequence (s1 == s2, idx1 != idx2)
 * - Negative inter: patches from different sequences (s1 != s2)
 *
 * Target indices (t1, t2):
 * - 0 = ref
 * - 1-5 = target set (e1-e5 for easy, h1-h5 for hard, t1-t5 for tough)
 */
struct VerificationTaskPair {
    std::string s1;  ///< First scene name
    int t1 = 0;      ///< First target index (0=ref, 1-5=target)
    int idx1 = 0;    ///< First patch index within stack
    std::string s2;  ///< Second scene name
    int t2 = 0;      ///< Second target index
    int idx2 = 0;    ///< Second patch index

    bool operator==(const VerificationTaskPair& other) const {
        return s1 == other.s1 && t1 == other.t1 && idx1 == other.idx1 &&
               s2 == other.s2 && t2 == other.t2 && idx2 == other.idx2;
    }
};

/**
 * @brief Retrieval task item from HPatches task files
 *
 * Represents a single patch for retrieval (query or distractor).
 * Always references the "ref" target at the given patch index.
 */
struct RetrievalTaskItem {
    std::string s;   ///< Scene name
    int idx = 0;     ///< Patch index within ref stack

    bool operator==(const RetrievalTaskItem& other) const {
        return s == other.s && idx == other.idx;
    }
};

/**
 * @brief Matching method for patch benchmark evaluation
 */
enum class PatchMatchingMethod {
    NEAREST_NEIGHBOR,  ///< 1-NN via cv::BFMatcher
    MANUAL_NN,         ///< 1-NN via manual row-by-row L2 distance loop
    RATIO_TEST         ///< Lowe's ratio test: reject if d1/d2 >= threshold
};

/**
 * @brief Configuration for descriptor matching in patch benchmark
 */
struct MatchingConfig {
    PatchMatchingMethod method = PatchMatchingMethod::NEAREST_NEIGHBOR;
    float ratio_threshold = 0.8f;  ///< Ratio test threshold (only for RATIO_TEST)
    int norm_type = cv::NORM_L2;   ///< OpenCV norm type (NORM_L2 or NORM_L1)
};

/**
 * @brief Configuration for patch benchmark
 */
struct Config {
    // Dataset
    std::string patches_dir = "../hpatches-release";
    std::vector<std::string> scenes;  ///< Empty = all scenes
    bool color = false;

    // Difficulty levels
    bool include_easy = true;
    bool include_hard = true;
    bool include_tough = true;

    // Tasks to run
    bool matching_enabled = true;
    MatchingConfig matching;
    bool verification_enabled = true;
    bool verification_same_seq = true;   ///< Same-sequence negatives
    bool verification_diff_seq = true;   ///< Different-sequence negatives
    bool retrieval_enabled = true;

    // Task parameters (HPatches paper protocol)
    int verification_num_positives = 200000;
    int verification_num_negatives = 1000000;
    int retrieval_num_queries = 10000;
    int retrieval_num_distractors = 20000;
    unsigned int random_seed = 1337;

    // Task file configuration
    std::string task_source = "csv";  ///< "csv", "db", or "random"
    std::string task_set = "hpatches_v1.1";
    std::string task_split = "full";  ///< "full", "illum", "view", "a", "b", "c"
    std::string tasks_dir;

    // Loaded task data
    std::vector<VerificationTaskPair> verification_pos_pairs;
    std::vector<VerificationTaskPair> verification_neg_intra_pairs;
    std::vector<VerificationTaskPair> verification_neg_inter_pairs;
    std::vector<RetrievalTaskItem> retrieval_queries;
    std::vector<RetrievalTaskItem> retrieval_distractors;

    // Descriptor caching
    bool use_cached_descriptors = false;
    bool store_descriptors_to_db = false;
    std::string descriptor_cache_name;
    int descriptor_cache_id = -1;

    // Output
    bool verbose = true;
    bool print_results = true;
    int num_threads = 4;
};

/**
 * @brief Results from patch benchmark evaluation
 */
struct Results {
    std::string descriptor_name;
    int descriptor_dimension = 0;

    // Matching results
    float mAP_overall = 0.0f;
    float accuracy_overall = 0.0f;
    float mAP_easy = 0.0f;
    float mAP_hard = 0.0f;
    float mAP_tough = 0.0f;
    float mAP_illumination = 0.0f;
    float mAP_viewpoint = 0.0f;
    float mAP_illumination_easy = 0.0f;
    float mAP_illumination_hard = 0.0f;
    float mAP_viewpoint_easy = 0.0f;
    float mAP_viewpoint_hard = 0.0f;

    // Verification same-seq results
    float verification_same_overall = 0.0f;
    float verification_same_easy = 0.0f;
    float verification_same_hard = 0.0f;
    float verification_same_tough = 0.0f;
    float verification_same_illumination = 0.0f;
    float verification_same_viewpoint = 0.0f;
    float verification_same_illumination_easy = 0.0f;
    float verification_same_illumination_hard = 0.0f;
    float verification_same_viewpoint_easy = 0.0f;
    float verification_same_viewpoint_hard = 0.0f;

    // Verification diff-seq results
    float verification_diff_overall = 0.0f;
    float verification_diff_easy = 0.0f;
    float verification_diff_hard = 0.0f;
    float verification_diff_tough = 0.0f;
    float verification_diff_illumination = 0.0f;
    float verification_diff_viewpoint = 0.0f;
    float verification_diff_illumination_easy = 0.0f;
    float verification_diff_illumination_hard = 0.0f;
    float verification_diff_viewpoint_easy = 0.0f;
    float verification_diff_viewpoint_hard = 0.0f;

    // Retrieval results
    float retrieval_overall = 0.0f;
    float retrieval_easy = 0.0f;
    float retrieval_hard = 0.0f;
    float retrieval_tough = 0.0f;
    float retrieval_illumination = 0.0f;
    float retrieval_viewpoint = 0.0f;
    float retrieval_illumination_easy = 0.0f;
    float retrieval_illumination_hard = 0.0f;
    float retrieval_viewpoint_easy = 0.0f;
    float retrieval_viewpoint_hard = 0.0f;

    // Statistics
    int num_scenes = 0;
    int num_patches = 0;
    double processing_time_ms = 0.0;
};

} // namespace thesis_project::benchmark
