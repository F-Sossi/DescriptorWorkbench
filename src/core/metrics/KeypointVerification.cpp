#include "KeypointVerification.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <omp.h>
#include <functional>

#include "thesis_project/logging.hpp"

namespace thesis_project::metrics {

    bool isCorrectMatch(const cv::Point2f& query_pt,
                        const cv::Point2f& candidate_pt,
                        const cv::Mat& homography,
                        const std::vector<cv::KeyPoint>& all_target_keypoints,
                        const double tolerance_px) {

        // Project query keypoint to target image
        std::vector<cv::Point2f> query_pts = {query_pt};
        std::vector<cv::Point2f> projected_pts;
        cv::perspectiveTransform(query_pts, projected_pts, homography);

        const cv::Point2f projected = projected_pts[0];

        // Check if projection is valid
        if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
            return false;
        }

        // Distance from projection to candidate
        double dist_to_candidate = cv::norm(projected - candidate_pt);

        // Must be within tolerance
        if (dist_to_candidate > tolerance_px) {
            return false;
        }

        // Check if candidate is closest to projection
        for (const auto& kp : all_target_keypoints) {
            if (const double dist = cv::norm(projected - kp.pt); dist < dist_to_candidate - 1e-6) {  // Small epsilon for floating point
                return false;  // Found a closer keypoint
            }
        }

        return true;
    }

    std::vector<std::string> sampleDistractorScenes(
        const std::string& query_scene,
        const std::vector<std::string>& all_scenes,
        const int num_distractors,
        const int seed) {

        std::vector<std::string> other_scenes;
        for (const auto& scene : all_scenes) {
            if (scene != query_scene) {
                other_scenes.push_back(scene);
            }
        }

        // Shuffle and take first num_distractors
        const std::size_t scene_hash = std::hash<std::string>{}(query_scene);
        const uint32_t combined_seed =
            static_cast<uint32_t>(seed) ^
            static_cast<uint32_t>(scene_hash) ^
            static_cast<uint32_t>(scene_hash >> 32);
        std::mt19937 rng(combined_seed);
        std::shuffle(other_scenes.begin(), other_scenes.end(), rng);

        const int count = std::min(num_distractors, static_cast<int>(other_scenes.size()));
        return std::vector<std::string>(other_scenes.begin(), other_scenes.begin() + count);
    }

    std::vector<VerificationCandidate> buildVerificationCandidates(
        const std::string& scene_name,
        const std::vector<cv::KeyPoint>& ref_keypoints,
        const cv::Mat& ref_descriptors,
        const std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>& scene_images,
        const std::map<std::string, cv::Mat>& scene_homographies,
        const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& all_scenes,
        const std::vector<std::string>& distractor_scene_names,
        int num_distractors_per_scene,
        int seed) {

        std::vector<VerificationCandidate> candidates;
        std::mt19937 rng(seed);

        // For each keypoint in reference image
        for (size_t query_idx = 0; query_idx < ref_keypoints.size(); ++query_idx) {
            const auto& query_kp = ref_keypoints[query_idx];
            cv::Mat query_desc = ref_descriptors.row(query_idx);

            // === PART 1: Same-sequence matches (positive + hard negative) ===
            for (const auto& [target_image, target_data] : scene_images) {
                if (target_image == "1.ppm") continue;  // Skip reference image

                const auto& target_keypoints = target_data.first;
                const cv::Mat& target_descriptors = target_data.second;

                // Get homography for this image
                auto hom_it = scene_homographies.find(target_image);
                if (hom_it == scene_homographies.end()) {
                    continue;  // Skip if no homography available
                }
                const cv::Mat& homography = hom_it->second;

                // Match query descriptor to all target descriptors
                for (size_t cand_idx = 0; cand_idx < target_keypoints.size(); ++cand_idx) {
                    cv::Mat cand_desc = target_descriptors.row(cand_idx);

                    // Compute descriptor distance (L2 norm)
                    double distance = cv::norm(query_desc, cand_desc, cv::NORM_L2);

                    // Determine label using homography
                    bool is_correct = isCorrectMatch(
                        query_kp.pt,
                        target_keypoints[cand_idx].pt,
                        homography,
                        target_keypoints,
                        3.0  // tolerance_px
                    );

                    VerificationCandidate candidate;
                    candidate.query_pt = query_kp.pt;
                    candidate.query_idx = query_idx;
                    candidate.candidate_pt = target_keypoints[cand_idx].pt;
                    candidate.candidate_idx = cand_idx;
                    candidate.descriptor_distance = distance;
                    candidate.query_scene = scene_name;
                    candidate.query_image = "1.ppm";
                    candidate.candidate_scene = scene_name;
                    candidate.candidate_image = target_image;
                    candidate.label = is_correct ? +1 : -1;
                    candidate.is_same_sequence = true;
                    candidate.is_distractor = false;

                    candidates.push_back(candidate);
                }
            }

            // === PART 2: Out-of-sequence distractors (guaranteed negative) ===
            for (const auto& dist_scene : distractor_scene_names) {
                const auto& dist_images = all_scenes.at(dist_scene);
                if (dist_scene == scene_name) continue;  // Skip same scene

                // Sample random image from distractor scene
                if (dist_images.empty()) continue;

                std::vector<std::string> image_names;
                for (const auto& [img_name, _] : dist_images) {
                    image_names.push_back(img_name);
                }

                std::uniform_int_distribution<> img_dist(0, image_names.size() - 1);
                std::string random_image = image_names[img_dist(rng)];

                const auto& dist_data = dist_images.at(random_image);
                const auto& dist_keypoints = dist_data.first;
                const cv::Mat& dist_descriptors = dist_data.second;

                // Sample random keypoints from distractor image
                std::vector<size_t> kp_indices(dist_keypoints.size());
                std::iota(kp_indices.begin(), kp_indices.end(), 0);
                std::shuffle(kp_indices.begin(), kp_indices.end(), rng);

                int num_to_sample = std::min(num_distractors_per_scene,
                                             static_cast<int>(dist_keypoints.size()));

                for (int i = 0; i < num_to_sample; ++i) {
                    size_t cand_idx = kp_indices[i];
                    cv::Mat cand_desc = dist_descriptors.row(cand_idx);

                    double distance = cv::norm(query_desc, cand_desc, cv::NORM_L2);

                    VerificationCandidate candidate;
                    candidate.query_pt = query_kp.pt;
                    candidate.query_idx = query_idx;
                    candidate.candidate_pt = dist_keypoints[cand_idx].pt;
                    candidate.candidate_idx = cand_idx;
                    candidate.descriptor_distance = distance;
                    candidate.query_scene = scene_name;
                    candidate.query_image = "1.ppm";
                    candidate.candidate_scene = dist_scene;
                    candidate.candidate_image = random_image;
                    candidate.label = -1;  // Always negative (distractor)
                    candidate.is_same_sequence = false;
                    candidate.is_distractor = true;

                    candidates.push_back(candidate);
                }
            }
        }

        return candidates;
    }

    VerificationResult evaluateVerification(const std::vector<VerificationCandidate>& candidates) {
        VerificationResult result;

        if (candidates.empty()) {
            return result;
        }

        // Sort candidates by descriptor distance (ascending = higher confidence)
        std::vector<VerificationCandidate> sorted_candidates = candidates;
        std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                  [](const VerificationCandidate& a, const VerificationCandidate& b) {
                      return a.descriptor_distance < b.descriptor_distance;
                  });

        // Build relevance vector
        std::vector<int> relevance;
        relevance.reserve(sorted_candidates.size());
        for (const auto& cand : sorted_candidates) {
            relevance.push_back(cand.label > 0 ? 1 : 0);
        }

        // Compute AP
        int total_relevant = std::accumulate(relevance.begin(), relevance.end(), 0);

        if (total_relevant == 0) {
            result.average_precision = 0.0;
            result.total_candidates = sorted_candidates.size();
            return result;
        }

        double ap_sum = 0.0;
        int hits = 0;

        for (size_t k = 0; k < relevance.size(); ++k) {
            if (relevance[k]) {
                hits++;
                double precision_at_k = static_cast<double>(hits) / static_cast<double>(k + 1);
                ap_sum += precision_at_k;
            }
        }

        result.average_precision = ap_sum / static_cast<double>(total_relevant);
        result.total_candidates = sorted_candidates.size();
        result.total_correct = total_relevant;

        // Count distractors
        for (const auto& cand : sorted_candidates) {
            if (cand.is_distractor) {
                result.total_distractors++;
            }
        }

        return result;
    }

    VerificationResult computeDatasetVerification(
        const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& scene_data,
        const std::map<std::string, std::map<std::string, cv::Mat>>& scene_homographies,
        int num_distractor_scenes,
        int num_distractors_per_scene,
        int seed) {

        VerificationResult aggregated_result;

        // Get list of all scenes for distractor sampling AND parallel indexing
        std::vector<std::string> all_scenes;
        for (const auto& [scene_name, _] : scene_data) {
            all_scenes.push_back(scene_name);
        }

        // Thread-safe result storage
        std::vector<double> scene_aps;
        std::vector<std::string> valid_scenes;

        // OpenMP parallel loop over scenes (each scene is independent)
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < all_scenes.size(); i++) {
            const auto& scene_name = all_scenes[i];
            const auto& scene_images = scene_data.at(scene_name);

            // Get reference image data
            auto ref_it = scene_images.find("1.ppm");
            if (ref_it == scene_images.end()) {
                continue;  // Skip scenes without reference image
            }

            const auto& ref_keypoints = ref_it->second.first;
            const cv::Mat& ref_descriptors = ref_it->second.second;

            // Get homographies for this scene
            auto hom_it = scene_homographies.find(scene_name);
            if (hom_it == scene_homographies.end()) {
                continue;  // Skip scenes without homographies
            }

            // Sample distractor scenes
            auto distractor_scene_names = sampleDistractorScenes(
                scene_name, all_scenes, num_distractor_scenes, seed);

            // Build verification candidates for this scene
            auto candidates = buildVerificationCandidates(
                scene_name,
                ref_keypoints,
                ref_descriptors,
                scene_images,
                hom_it->second,
                scene_data,
                distractor_scene_names,
                num_distractors_per_scene,
                seed
            );

            // Evaluate verification for this scene
            auto scene_result = evaluateVerification(candidates);

            // Thread-safe aggregation of results
            if (scene_result.total_candidates > 0) {
                #pragma omp critical
                {
                    scene_aps.push_back(scene_result.average_precision);
                    valid_scenes.push_back(scene_name);
                    aggregated_result.per_scene_ap[scene_name] = scene_result.average_precision;
                    aggregated_result.total_queries += scene_result.total_queries;
                    aggregated_result.total_candidates += scene_result.total_candidates;
                    aggregated_result.total_distractors += scene_result.total_distractors;
                    aggregated_result.total_correct += scene_result.total_correct;

                    // Category breakdown (viewpoint vs illumination)
                    if (!scene_name.empty() && scene_name[0] == 'v') {
                        // Viewpoint scene
                        aggregated_result.viewpoint_ap += scene_result.average_precision;
                    } else if (!scene_name.empty() && scene_name[0] == 'i') {
                        // Illumination scene
                        aggregated_result.illumination_ap += scene_result.average_precision;
                    }
                }
            }
        }

        // Compute overall average precision
        if (!scene_aps.empty()) {
            aggregated_result.average_precision = std::accumulate(scene_aps.begin(), scene_aps.end(), 0.0) / scene_aps.size();

            // Normalize category APs by scene count
            int viewpoint_count = 0;
            int illumination_count = 0;
            for (const auto& [scene_name, _] : aggregated_result.per_scene_ap) {
                if (!scene_name.empty() && scene_name[0] == 'v') viewpoint_count++;
                else if (!scene_name.empty() && scene_name[0] == 'i') illumination_count++;
            }

            if (viewpoint_count > 0) {
                aggregated_result.viewpoint_ap /= viewpoint_count;
            }
            if (illumination_count > 0) {
                aggregated_result.illumination_ap /= illumination_count;
            }
        }

        return aggregated_result;
    }

// ============================================================================
// KEYPOINT RETRIEVAL IMPLEMENTATION (Three-Tier Labeling)
// ============================================================================

    CandidateLabel assignRetrievalLabel(bool is_same_sequence, bool is_closest_to_projection) {
        if (!is_same_sequence) {
            return CandidateLabel::DISTRACTOR;  // y = -1 (out-of-sequence)
        }

        if (is_closest_to_projection) {
            return CandidateLabel::TRUE_POSITIVE;  // y = +1 (in-sequence AND closest)
        }

        return CandidateLabel::HARD_NEGATIVE;  // y = 0 (in-sequence but NOT closest)
    }

    std::vector<RetrievalCandidate> buildRetrievalCandidates(
        const std::string& scene_name,
        const std::vector<cv::KeyPoint>& ref_keypoints,
        const cv::Mat& ref_descriptors,
        const std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>& scene_images,
        const std::map<std::string, cv::Mat>& scene_homographies,
        const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& distractor_scenes,
        int num_distractors_per_scene,
        int seed) {

        std::vector<RetrievalCandidate> candidates;
        std::mt19937 rng(seed);

        // For each keypoint in reference image
        for (size_t query_idx = 0; query_idx < ref_keypoints.size(); ++query_idx) {
            const auto& query_kp = ref_keypoints[query_idx];
            cv::Mat query_desc = ref_descriptors.row(query_idx);

            // === PART 1: Same-sequence matches (y=+1 or y=0) ===
            for (const auto& [target_image, target_data] : scene_images) {
                if (target_image == "1.ppm") continue;  // Skip reference image

                const auto& target_keypoints = target_data.first;
                const cv::Mat& target_descriptors = target_data.second;

                // Get homography for this image
                auto hom_it = scene_homographies.find(target_image);
                if (hom_it == scene_homographies.end()) {
                    continue;  // Skip if no homography available
                }
                const cv::Mat& homography = hom_it->second;

                // Match query descriptor to all target descriptors
                for (size_t cand_idx = 0; cand_idx < target_keypoints.size(); ++cand_idx) {
                    cv::Mat cand_desc = target_descriptors.row(cand_idx);

                    // Compute descriptor distance (L2 norm)
                    double distance = cv::norm(query_desc, cand_desc, cv::NORM_L2);

                    // Determine if this is geometrically correct using homography
                    bool is_correct = isCorrectMatch(
                        query_kp.pt,
                        target_keypoints[cand_idx].pt,
                        homography,
                        target_keypoints,
                        3.0  // tolerance_px
                    );

                    // Assign three-tier label
                    CandidateLabel label = assignRetrievalLabel(
                        true,  // is_same_sequence
                        is_correct  // is_closest_to_projection
                    );

                    RetrievalCandidate candidate;
                    candidate.query_pt = query_kp.pt;
                    candidate.query_idx = query_idx;
                    candidate.candidate_pt = target_keypoints[cand_idx].pt;
                    candidate.candidate_idx = cand_idx;
                    candidate.descriptor_distance = distance;
                    candidate.query_scene = scene_name;
                    candidate.query_image = "1.ppm";
                    candidate.candidate_scene = scene_name;
                    candidate.candidate_image = target_image;
                    candidate.label = label;
                    candidate.is_same_sequence = true;
                    candidate.is_distractor = false;

                    candidates.push_back(candidate);
                }
            }

            // === PART 2: Out-of-sequence distractors (guaranteed y=-1) ===
            for (const auto& [dist_scene, dist_images] : distractor_scenes) {
                // Sample random image from distractor scene
                std::vector<std::string> image_names;
                for (const auto& [img_name, _] : dist_images) {
                    image_names.push_back(img_name);
                }

                if (image_names.empty()) continue;

                std::uniform_int_distribution<> img_dist(0, image_names.size() - 1);
                std::string random_image = image_names[img_dist(rng)];

                const auto& dist_keypoints = dist_images.at(random_image).first;
                const cv::Mat& dist_descriptors = dist_images.at(random_image).second;

                // Sample random keypoints from distractor image
                std::vector<size_t> kp_indices(dist_keypoints.size());
                std::iota(kp_indices.begin(), kp_indices.end(), 0);
                std::shuffle(kp_indices.begin(), kp_indices.end(), rng);

                int num_to_sample = std::min(num_distractors_per_scene,
                                            static_cast<int>(dist_keypoints.size()));

                for (int i = 0; i < num_to_sample; ++i) {
                    size_t cand_idx = kp_indices[i];
                    cv::Mat cand_desc = dist_descriptors.row(cand_idx);

                    double distance = cv::norm(query_desc, cand_desc, cv::NORM_L2);

                    RetrievalCandidate candidate;
                    candidate.query_pt = query_kp.pt;
                    candidate.query_idx = query_idx;
                    candidate.candidate_pt = dist_keypoints[cand_idx].pt;
                    candidate.candidate_idx = cand_idx;
                    candidate.descriptor_distance = distance;
                    candidate.query_scene = scene_name;
                    candidate.query_image = "1.ppm";
                    candidate.candidate_scene = dist_scene;
                    candidate.candidate_image = random_image;
                    candidate.label = CandidateLabel::DISTRACTOR;  // Always y=-1
                    candidate.is_same_sequence = false;
                    candidate.is_distractor = true;

                    candidates.push_back(candidate);
                }
            }
        }

        return candidates;
    }

    RetrievalResult evaluateRetrieval(const std::vector<RetrievalCandidate>& candidates) {
        RetrievalResult result;

        if (candidates.empty()) {
            return result;
        }

        // Make a copy for sorting
        std::vector<RetrievalCandidate> sorted_candidates = candidates;

        // Sort by descriptor distance (ascending = higher confidence)
        std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                  [](const RetrievalCandidate& a, const RetrievalCandidate& b) {
                      return a.descriptor_distance < b.descriptor_distance;
                  });

        // Build relevance vector (only y=+1 is relevant)
        std::vector<int> relevance;
        relevance.reserve(sorted_candidates.size());
        for (const auto& cand : sorted_candidates) {
            relevance.push_back(cand.label == CandidateLabel::TRUE_POSITIVE ? 1 : 0);
        }

        // Count label distribution
        for (const auto& cand : sorted_candidates) {
            if (cand.label == CandidateLabel::TRUE_POSITIVE) {
                result.num_true_positives++;
            } else if (cand.label == CandidateLabel::HARD_NEGATIVE) {
                result.num_hard_negatives++;
            } else {
                result.num_distractors++;
            }
        }

        // Compute AP (standard IR formula)
        const int total_relevant = std::accumulate(relevance.begin(), relevance.end(), 0);

        if (total_relevant == 0) {
            result.average_precision = 0.0;
            result.total_candidates = sorted_candidates.size();
            return result;
        }

        double ap_sum = 0.0;
        int hits = 0;

        for (size_t k = 0; k < relevance.size(); ++k) {
            if (relevance[k]) {
                hits++;
                double precision_at_k = static_cast<double>(hits) / static_cast<double>(k + 1);
                ap_sum += precision_at_k;
            }
        }

        result.average_precision = ap_sum / static_cast<double>(total_relevant);
        result.total_candidates = sorted_candidates.size();
        result.total_queries = total_relevant;

        return result;
    }

    RetrievalDiagnostics analyzeRetrievalPerformance(
        const std::vector<RetrievalCandidate>& candidates,
        int k) {

        RetrievalDiagnostics diag;

        if (candidates.empty()) {
            return diag;
        }

        // Compute AP
        auto result = evaluateRetrieval(candidates);
        diag.ap = result.average_precision;
        diag.total_queries = result.total_candidates;
        diag.true_positives = result.num_true_positives;
        diag.hard_negatives = result.num_hard_negatives;
        diag.distractors = result.num_distractors;

        // Make a copy for sorting
        std::vector<RetrievalCandidate> sorted_candidates = candidates;
        std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                  [](const RetrievalCandidate& a, const RetrievalCandidate& b) {
                      return a.descriptor_distance < b.descriptor_distance;
                  });

        // Analyze top-k
        int top_k = std::min(k, static_cast<int>(sorted_candidates.size()));
        for (int i = 0; i < top_k; ++i) {
            if (sorted_candidates[i].label == CandidateLabel::TRUE_POSITIVE) {
                diag.tp_in_top_k++;
            } else if (sorted_candidates[i].label == CandidateLabel::HARD_NEGATIVE) {
                diag.hn_in_top_k++;
            } else {
                diag.dist_in_top_k++;
            }
        }

        return diag;
    }

    RetrievalResult computeDatasetRetrieval(
        const std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>>& scene_data,
        const std::map<std::string, std::map<std::string, cv::Mat>>& scene_homographies,
        int num_distractor_scenes,
        int num_distractors_per_scene,
        int seed) {

        RetrievalResult aggregated_result;
        LOG_INFO("Retrieval task: scene_data size=" + std::to_string(scene_data.size()) +
                 ", homography size=" + std::to_string(scene_homographies.size()));

        // Get list of all scenes for distractor sampling AND parallel indexing
        std::vector<std::string> all_scenes;
        for (const auto& [scene_name, _] : scene_data) {
            all_scenes.push_back(scene_name);
        }

        // Thread-safe result storage
        std::vector<double> scene_aps;
        std::vector<std::string> valid_scenes;

        // OpenMP parallel loop over scenes (each scene is independent)
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < all_scenes.size(); i++) {
            const auto& scene_name = all_scenes[i];
            const auto& scene_images = scene_data.at(scene_name);
            LOG_DEBUG("Retrieval scene " + scene_name + " has " + std::to_string(scene_images.size()) + " images");

            // Get reference image data
            auto ref_it = scene_images.find("1.ppm");
            if (ref_it == scene_images.end()) {
                LOG_WARNING("Retrieval: missing reference image for scene " + scene_name);
                continue;  // Skip scenes without reference image
            }

            const auto& ref_keypoints = ref_it->second.first;
            const cv::Mat& ref_descriptors = ref_it->second.second;

            // Get homographies for this scene
            auto hom_it = scene_homographies.find(scene_name);
            if (hom_it == scene_homographies.end()) {
                LOG_WARNING("Retrieval: missing homographies for scene " + scene_name);
                continue;  // Skip scenes without homographies
            }

            // Sample distractor scenes
            auto distractor_scene_names = sampleDistractorScenes(
                scene_name, all_scenes, num_distractor_scenes, seed);

            // Build distractor scene data map
            std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>> distractor_data;
            for (const auto& dist_scene : distractor_scene_names) {
                distractor_data[dist_scene] = scene_data.at(dist_scene);
            }

            // Build retrieval candidates for this scene
            auto candidates = buildRetrievalCandidates(
                scene_name,
                ref_keypoints,
                ref_descriptors,
                scene_images,
                hom_it->second,
                distractor_data,
                num_distractors_per_scene,
                seed
            );

            LOG_DEBUG("Retrieval: scene " + scene_name + " produced " +
                      std::to_string(candidates.size()) + " candidates");

            // Evaluate retrieval for this scene
            auto scene_result = evaluateRetrieval(candidates);
            LOG_DEBUG("Retrieval: scene " + scene_name + " AP=" + std::to_string(scene_result.average_precision) +
                      " TP=" + std::to_string(scene_result.num_true_positives) +
                      " HN=" + std::to_string(scene_result.num_hard_negatives) +
                      " D=" + std::to_string(scene_result.num_distractors));

            // Thread-safe aggregation of results
            if (scene_result.total_candidates > 0) {
                #pragma omp critical
                {
                    scene_aps.push_back(scene_result.average_precision);
                    valid_scenes.push_back(scene_name);
                    aggregated_result.per_scene_ap[scene_name] = scene_result.average_precision;
                    aggregated_result.total_queries += scene_result.total_queries;
                    aggregated_result.total_candidates += scene_result.total_candidates;
                    aggregated_result.num_true_positives += scene_result.num_true_positives;
                    aggregated_result.num_hard_negatives += scene_result.num_hard_negatives;
                    aggregated_result.num_distractors += scene_result.num_distractors;

                    // Category breakdown (viewpoint vs illumination)
                    if (!scene_name.empty() && scene_name[0] == 'v') {
                        // Viewpoint scene
                        aggregated_result.viewpoint_ap += scene_result.average_precision;
                    } else if (!scene_name.empty() && scene_name[0] == 'i') {
                        // Illumination scene
                        aggregated_result.illumination_ap += scene_result.average_precision;
                    }
                }
            }
        }

        // Compute overall average precision
        if (!scene_aps.empty()) {
            aggregated_result.average_precision = std::accumulate(scene_aps.begin(), scene_aps.end(), 0.0) / scene_aps.size();

            // Normalize category APs by scene count
            int viewpoint_count = 0;
            int illumination_count = 0;
            for (const auto& [scene_name, _] : aggregated_result.per_scene_ap) {
                if (!scene_name.empty() && scene_name[0] == 'v') viewpoint_count++;
                else if (!scene_name.empty() && scene_name[0] == 'i') illumination_count++;
            }

            if (viewpoint_count > 0) {
                aggregated_result.viewpoint_ap /= viewpoint_count;
            }
            if (illumination_count > 0) {
                aggregated_result.illumination_ap /= illumination_count;
            }
        }

        LOG_INFO("Retrieval aggregate: AP=" + std::to_string(aggregated_result.average_precision) +
                 " scenes=" + std::to_string(aggregated_result.per_scene_ap.size()) +
                 " total_candidates=" + std::to_string(aggregated_result.total_candidates) +
                 " TP=" + std::to_string(aggregated_result.num_true_positives) +
                 " HN=" + std::to_string(aggregated_result.num_hard_negatives) +
                 " D=" + std::to_string(aggregated_result.num_distractors));

        return aggregated_result;
    }

}
