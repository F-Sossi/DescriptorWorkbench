#ifndef CORE_METRICS_KEYPOINT_VERIFICATION_HPP
#define CORE_METRICS_KEYPOINT_VERIFICATION_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <map>

namespace thesis_project::metrics {

/**
 * @brief Represents a candidate match for keypoint verification
 *
 * Based on Bojanic et al. (2020) Equation 1-2:
 * - Matches from same sequence (potential true positives)
 * - Matches from different sequences (guaranteed false positives/distractors)
 */
struct VerificationCandidate {
    // Query keypoint from reference image
    cv::Point2f query_pt;
    int query_idx;

    // Candidate match keypoint
    cv::Point2f candidate_pt;
    int candidate_idx;

    // Descriptor distance (used for ranking)
    double descriptor_distance;

    // Scene and image identifiers
    std::string query_scene;
    std::string query_image;
    std::string candidate_scene;
    std::string candidate_image;

    // Ground truth label (Equation 2)
    // +1: in-sequence AND closest to homography projection
    // -1: otherwise (either out-of-sequence OR in-sequence but incorrect)
    int label;

    // Metadata
    bool is_same_sequence;  // True if query_scene == candidate_scene
    bool is_distractor;     // True if out-of-sequence (guaranteed incorrect)
};

/**
 * @brief Result of verification task for all queries
 */
struct VerificationResult {
    double average_precision = 0.0;           // AP over all verification candidates
    int total_queries = 0;                    // Number of keypoints verified
    int total_candidates = 0;                 // Total matches considered
    int total_distractors = 0;                // Number of out-of-sequence matches
    int total_correct = 0;                    // Number of true positives (label=+1)

    // Per-scene breakdown
    std::map<std::string, double> per_scene_ap;

    // Category breakdown
    double viewpoint_ap = 0.0;
    double illumination_ap = 0.0;
};

/**
 * @brief Determine if candidate is correct match using homography
 *
 * Based on Equation 2: y = +1 if d(H·x, x') <= d(H·x, x'') for all x'' in target
 *
 * @param query_pt Query keypoint location
 * @param candidate_pt Candidate match location
 * @param homography Homography matrix from query to candidate image
 * @param all_target_keypoints All keypoints in target image
 * @param tolerance_px Maximum distance tolerance (default 3.0px)
 * @return true if candidate is the closest match to projected query point
 */
bool isCorrectMatch(const cv::Point2f& query_pt,
                   const cv::Point2f& candidate_pt,
                   const cv::Mat& homography,
                   const std::vector<cv::KeyPoint>& all_target_keypoints,
                   double tolerance_px = 3.0);

/**
 * @brief Sample random images from other sequences as distractors
 *
 * @param query_scene Current scene name
 * @param all_scenes List of all available scenes
 * @param num_distractors Number of distractor scenes to sample
 * @param seed Random seed for reproducibility
 * @return List of sampled distractor scene names
 */
std::vector<std::string> sampleDistractorScenes(
    const std::string& query_scene,
    const std::vector<std::string>& all_scenes,
    int num_distractors,
    int seed = 42);

/**
 * @brief Build verification candidate set for a single scene
 *
 * For each keypoint in reference image:
 * 1. Find matches in same-sequence images (positive + hard negative)
 * 2. Find matches in random other-sequence images (distractors)
 * 3. Rank by descriptor distance
 * 4. Label using homography ground truth
 *
 * @param scene_name Scene name
 * @param ref_keypoints Keypoints from reference image (1.ppm)
 * @param ref_descriptors Descriptors from reference image
 * @param scene_images Map of image names to keypoints/descriptors for this scene
 * @param scene_homographies Map of image names to homographies (H_1_to_img)
 * @param distractor_scenes Map of distractor scenes to their image data
 * @param num_distractors_per_scene Number of distractor images to sample per scene
 * @param seed Random seed for reproducibility
 * @return List of verification candidates with labels
 */
std::vector<VerificationCandidate> buildVerificationCandidates(
    const std::string& scene_name,
    const std::vector<cv::KeyPoint>& ref_keypoints,
    const cv::Mat& ref_descriptors,
    const std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>& scene_images,
    const std::map<std::string, cv::Mat>& scene_homographies,
    const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& distractor_scenes,
    int num_distractors_per_scene = 20,
    int seed = 42);

/**
 * @brief Evaluate verification task using Average Precision
 *
 * @param candidates List of verification candidates (sorted by descriptor distance)
 * @return Verification result with AP and metadata
 */
VerificationResult evaluateVerification(const std::vector<VerificationCandidate>& candidates);

/**
 * @brief Compute verification metrics for entire dataset
 *
 * @param scene_data Map of scene names to their keypoints/descriptors/homographies
 * @param num_distractor_scenes Number of distractor scenes to sample per query scene
 * @param num_distractors_per_scene Number of distractor images per scene
 * @param seed Random seed for reproducibility
 * @return Aggregated verification result
 */
VerificationResult computeDatasetVerification(
    const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& scene_data,
    const std::map<std::string, std::map<std::string, cv::Mat>>& scene_homographies,
    int num_distractor_scenes = 10,
    int num_distractors_per_scene = 20,
    int seed = 42);

// ============================================================================
// KEYPOINT RETRIEVAL (Three-Tier Labeling)
// ============================================================================

/**
 * @brief Three-tier label for keypoint retrieval (Bojanic et al. Equation 6)
 */
enum class CandidateLabel {
    TRUE_POSITIVE = 1,      // In-sequence AND closest to H·x (y=+1)
    HARD_NEGATIVE = 0,      // In-sequence but NOT closest (y=0)
    DISTRACTOR = -1         // Out-of-sequence (y=-1)
};

/**
 * @brief Represents a candidate match for keypoint retrieval with three-tier labeling
 *
 * Based on Bojanic et al. (2020) Equation 5-6:
 * - y = +1: In-sequence AND closest to homography projection (true positive)
 * - y = 0: In-sequence but NOT closest (hard negative)
 * - y = -1: Out-of-sequence (distractor)
 */
struct RetrievalCandidate {
    // Query keypoint from reference image
    cv::Point2f query_pt;
    int query_idx;

    // Candidate match keypoint
    cv::Point2f candidate_pt;
    int candidate_idx;

    // Descriptor distance (used for ranking)
    double descriptor_distance;

    // Scene and image identifiers
    std::string query_scene;
    std::string query_image;
    std::string candidate_scene;
    std::string candidate_image;

    // Three-tier ground truth label (Equation 6)
    CandidateLabel label;

    // Metadata
    bool is_same_sequence;  // True if query_scene == candidate_scene
    bool is_distractor;     // True if out-of-sequence (guaranteed y=-1)
};

/**
 * @brief Result of retrieval task for all queries
 */
struct RetrievalResult {
    double average_precision = 0.0;           // AP over all retrieval candidates
    int total_queries = 0;                    // Number of keypoints queried (with y=+1)
    int total_candidates = 0;                 // Total matches considered

    // Label distribution
    int num_true_positives = 0;               // Candidates with y=+1
    int num_hard_negatives = 0;               // Candidates with y=0
    int num_distractors = 0;                  // Candidates with y=-1

    // Per-scene breakdown
    std::map<std::string, double> per_scene_ap;

    // Category breakdown
    double viewpoint_ap = 0.0;
    double illumination_ap = 0.0;
};

/**
 * @brief Diagnostic information for retrieval performance
 */
struct RetrievalDiagnostics {
    double ap = 0.0;
    int total_queries = 0;
    int true_positives = 0;     // y = +1
    int hard_negatives = 0;     // y = 0
    int distractors = 0;        // y = -1

    // Top-k analysis
    int tp_in_top_k = 0;        // True positives ranked in top-k
    int hn_in_top_k = 0;        // Hard negatives ranked in top-k
    int dist_in_top_k = 0;      // Distractors ranked in top-k
};

/**
 * @brief Assign three-tier label based on sequence membership and geometric correctness
 *
 * Based on Bojanic et al. Equation 6:
 * - y = +1: In-sequence AND closest to H·x
 * - y = 0: In-sequence but NOT closest
 * - y = -1: Out-of-sequence
 *
 * @param is_same_sequence Whether candidate is from same sequence as query
 * @param is_closest_to_projection Whether candidate is geometrically correct
 * @return Three-tier label
 */
CandidateLabel assignRetrievalLabel(bool is_same_sequence, bool is_closest_to_projection);

/**
 * @brief Build retrieval candidate set for a single scene with three-tier labels
 *
 * For each keypoint in reference image:
 * 1. Find matches in same-sequence images (y=+1 or y=0)
 * 2. Find matches in random other-sequence images (y=-1)
 * 3. Rank by descriptor distance
 * 4. Label using three-tier system (Equation 6)
 *
 * @param scene_name Scene name
 * @param ref_keypoints Keypoints from reference image (1.ppm)
 * @param ref_descriptors Descriptors from reference image
 * @param scene_images Map of image names to keypoints/descriptors for this scene
 * @param scene_homographies Map of image names to homographies (H_1_to_img)
 * @param distractor_scenes Map of distractor scenes to their image data
 * @param num_distractors_per_scene Number of distractor images to sample per scene
 * @param seed Random seed for reproducibility
 * @return List of retrieval candidates with three-tier labels
 */
std::vector<RetrievalCandidate> buildRetrievalCandidates(
    const std::string& scene_name,
    const std::vector<cv::KeyPoint>& ref_keypoints,
    const cv::Mat& ref_descriptors,
    const std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>& scene_images,
    const std::map<std::string, cv::Mat>& scene_homographies,
    const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& distractor_scenes,
    int num_distractors_per_scene = 20,
    int seed = 42);

/**
 * @brief Evaluate retrieval task using Average Precision
 *
 * Only y=+1 labels are treated as relevant for AP computation (standard IR practice)
 *
 * @param candidates List of retrieval candidates (sorted by descriptor distance)
 * @return Retrieval result with AP and label distribution
 */
RetrievalResult evaluateRetrieval(const std::vector<RetrievalCandidate>& candidates);

/**
 * @brief Analyze retrieval performance with detailed diagnostics
 *
 * @param candidates List of retrieval candidates
 * @param k Number of top results to analyze (default: 10)
 * @return Diagnostic information
 */
RetrievalDiagnostics analyzeRetrievalPerformance(
    const std::vector<RetrievalCandidate>& candidates,
    int k = 10);

/**
 * @brief Compute retrieval metrics for entire dataset
 *
 * @param scene_data Map of scene names to their keypoints/descriptors/homographies
 * @param num_distractor_scenes Number of distractor scenes to sample per query scene
 * @param num_distractors_per_scene Number of distractor images per scene
 * @param seed Random seed for reproducibility
 * @return Aggregated retrieval result
 */
RetrievalResult computeDatasetRetrieval(
    const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& scene_data,
    const std::map<std::string, std::map<std::string, cv::Mat>>& scene_homographies,
    int num_distractor_scenes = 10,
    int num_distractors_per_scene = 20,
    int seed = 42);

} // namespace thesis_project::metrics


#endif // CORE_METRICS_KEYPOINT_VERIFICATION_HPP
