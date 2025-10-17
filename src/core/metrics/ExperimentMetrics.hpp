#ifndef CORE_METRICS_EXPERIMENT_METRICS_HPP
#define CORE_METRICS_EXPERIMENT_METRICS_HPP

#include <string>
#include <vector>
#include <map>
#include <numeric>
#include "TrueAveragePrecision.hpp"

/**
 * @brief Structure to hold comprehensive experiment results metrics
 * 
 * This structure encapsulates all metrics computation and aggregation functionality
 * previously scattered throughout the image processing pipeline.
 */
struct ExperimentMetrics {
    // Per-image precision values
    std::vector<double> precisions_per_image;
    
    // Aggregate metrics  
    double mean_precision = 0.0;                        // Simple mean of per-image precisions (legacy)
    double legacy_macro_precision_by_scene = 0.0;       // Legacy: macro average of per-scene precisions (NOT IR-style mAP)
    double true_map_micro = 0.0;                        // TRUE IR-style mAP: micro average over queries with GT (conditional)
    double true_map_macro_by_scene = 0.0;               // TRUE IR-style mAP: macro average across scenes (conditional)
    double true_map_micro_including_zeros = 0.0;        // TRUE IR-style mAP: includes R=0 queries as AP=0 (punitive)
    double true_map_macro_by_scene_including_zeros = 0.0; // TRUE IR-style mAP: macro including R=0 (punitive)
    
    // Precision@K and Recall@K metrics  
    double precision_at_1 = 0.0;                        // Precision@1: % of queries where top result is correct
    double precision_at_5 = 0.0;                        // Precision@5: % of correct results in top-5  
    double precision_at_10 = 0.0;                       // Precision@10: % of correct results in top-10
    double recall_at_1 = 0.0;                          // Recall@1: % of true matches found in top-1 (same as P@1 for R=1)
    double recall_at_5 = 0.0;                          // Recall@5: % of true matches found in top-5
    double recall_at_10 = 0.0;                         // Recall@10: % of true matches found in top-10
    
    // Count metrics
    int total_matches = 0;
    int total_keypoints = 0;
    int total_images_processed = 0;
    
    // Per-scene breakdown (e.g., "i_dome", "v_wall")
    std::map<std::string, std::vector<double>> per_scene_precisions; // Store all precisions for proper averaging
    std::map<std::string, int> per_scene_matches;
    std::map<std::string, int> per_scene_keypoints;
    std::map<std::string, int> per_scene_image_count; // Track number of images per scene
    
    // True IR-style mAP storage
    std::vector<double> ap_per_query;                               // All query APs (micro mAP calculation)
    std::map<std::string, std::vector<double>> per_scene_ap;        // Per-scene AP values (macro mAP calculation)
    std::map<std::string, int> per_scene_excluded;                 // Count R=0 queries per scene
    int total_queries_processed = 0;                               // Total queries with potential matches
    int total_queries_excluded = 0;                                // Queries excluded (R=0)

    // Image-retrieval MAP storage
    std::vector<double> image_retrieval_ap_per_query;
    std::map<std::string, std::vector<double>> image_retrieval_ap_per_scene;
    double image_retrieval_map = -1.0;
    int image_retrieval_queries = 0;
    
    // Rank data for Precision@K/Recall@K calculation
    std::vector<int> ranks_per_query;                              // Rank of true match for each query (1-based, -1 if R=0)
    
    // Processing metadata
    double processing_time_ms = 0.0;
    bool success = true;  // Default to success - set false only on error
    std::string error_message;
    
    /**
     * @brief Calculate mean precision and macro average precision by scene
     * 
     * Mean Precision: Simple arithmetic mean of all per-image precisions
     * Mean Average Precision: Macro average of per-scene mean precisions (balanced across scenes)
     * 
     * WARNING: mean_average_precision is NOT Information Retrieval style mAP!
     * It's just a macro average of scene precisions for balanced evaluation.
     */
    void calculateMeanPrecision() {
        // Reset to avoid stale values
        mean_precision = 0.0;
        legacy_macro_precision_by_scene = 0.0;
        true_map_micro = 0.0;
        true_map_macro_by_scene = 0.0;
        true_map_micro_including_zeros = 0.0;
        true_map_macro_by_scene_including_zeros = 0.0;
        image_retrieval_map = -1.0;
        image_retrieval_queries = static_cast<int>(image_retrieval_ap_per_query.size());
        
        // Reset Precision@K and Recall@K metrics
        precision_at_1 = 0.0;
        precision_at_5 = 0.0;
        precision_at_10 = 0.0;
        recall_at_1 = 0.0;
        recall_at_5 = 0.0;
        recall_at_10 = 0.0;

        if (!precisions_per_image.empty()) {
            // Calculate simple mean precision across all images
            double sum = 0.0;
            for (double p : precisions_per_image) {
                sum += p;
            }
            mean_precision = sum / static_cast<double>(precisions_per_image.size());
        }

        if (!per_scene_precisions.empty()) {
            double scene_precision_sum = 0.0;
            int scene_count = 0;

            for (const auto& kv : per_scene_precisions) {
                const auto& precisions = kv.second;
                if (precisions.empty()) continue;
                
                double scene_sum = 0.0;
                for (double p : precisions) {
                    scene_sum += p;
                }
                scene_precision_sum += (scene_sum / static_cast<double>(precisions.size()));
                ++scene_count;
            }
            
            if (scene_count > 0) {
                legacy_macro_precision_by_scene = scene_precision_sum / static_cast<double>(scene_count);
            } else {
                legacy_macro_precision_by_scene = mean_precision; // fallback
            }
        } else {
            legacy_macro_precision_by_scene = mean_precision; // fallback if no scene breakdown
        }
        
        // Calculate true IR-style mAP metrics
        if (!ap_per_query.empty()) {
            // Micro mAP: average over all queries
            double ap_sum = 0.0;
            for (double ap : ap_per_query) {
                ap_sum += ap;
            }
            true_map_micro = ap_sum / static_cast<double>(ap_per_query.size());
        }
        
        if (!per_scene_ap.empty()) {
            // Macro mAP: average of per-scene mAPs
            double scene_map_sum = 0.0;
            int scene_count = 0;
            
            for (const auto& kv : per_scene_ap) {
                const auto& scene_aps = kv.second;
                if (scene_aps.empty()) continue;
                
                double scene_ap_sum = 0.0;
                for (double ap : scene_aps) {
                    scene_ap_sum += ap;
                }
                double scene_map = scene_ap_sum / static_cast<double>(scene_aps.size());
                scene_map_sum += scene_map;
                scene_count++;
            }
            
            if (scene_count > 0) {
                true_map_macro_by_scene = scene_map_sum / static_cast<double>(scene_count);
            }
        } else {
            true_map_macro_by_scene = true_map_micro; // fallback
        }
        
        // Calculate punitive metrics (including R=0 queries as AP=0)
        const int total_all_queries = total_queries_processed + total_queries_excluded;
        if (total_all_queries > 0) {
            // Micro including zeros: sum of APs / total queries (including R=0)
            double ap_sum = std::accumulate(ap_per_query.begin(), ap_per_query.end(), 0.0);
            true_map_micro_including_zeros = ap_sum / static_cast<double>(total_all_queries);
        }
        
        if (!per_scene_ap.empty()) {
            // Macro including zeros: average scene mAP where each scene includes R=0 queries
            double sum_scene_all = 0.0;
            int scene_cnt_all = 0;
            
            for (const auto& [scene, aps] : per_scene_ap) {
                int excluded_count = per_scene_excluded.count(scene) ? per_scene_excluded.at(scene) : 0;
                int total_scene_queries = static_cast<int>(aps.size()) + excluded_count;
                
                if (total_scene_queries == 0) continue; // Empty scene
                
                double scene_ap_sum = std::accumulate(aps.begin(), aps.end(), 0.0);
                double scene_map = scene_ap_sum / static_cast<double>(total_scene_queries);
                sum_scene_all += scene_map;
                scene_cnt_all++;
            }
            
            if (scene_cnt_all > 0) {
                true_map_macro_by_scene_including_zeros = sum_scene_all / static_cast<double>(scene_cnt_all);
            }
        } else {
            true_map_macro_by_scene_including_zeros = true_map_micro_including_zeros; // fallback
        }
        
        // Calculate Precision@K and Recall@K from rank data
        if (!ranks_per_query.empty()) {
            int queries_with_ranks = 0;
            int hits_at_1 = 0, hits_at_5 = 0, hits_at_10 = 0;
            
            // DEBUG: Count rank distribution
            std::map<int, int> rank_histogram;
            
            for (int rank : ranks_per_query) {
                if (rank > 0) { // Valid rank (not R=0)
                    queries_with_ranks++;
                    rank_histogram[rank]++;
                    if (rank <= 1) hits_at_1++;
                    if (rank <= 5) hits_at_5++;
                    if (rank <= 10) hits_at_10++;
                } else {
                    rank_histogram[-1]++; // R=0 queries
                }
            }
            
            // DEBUG: Print rank distribution for first few ranks
            /*std::cout << "[DEBUG] Rank distribution - Total queries: " << ranks_per_query.size()
                      << ", Valid ranks: " << queries_with_ranks << std::endl;
            std::cout << "[DEBUG] Ranks 1-10: ";*/
            for (int r = 1; r <= 10; r++) {
                std::cout << "R" << r << ":" << rank_histogram[r] << " ";
            }
            std::cout << " R=0:" << rank_histogram[-1] << std::endl;
            
            if (queries_with_ranks > 0) {
                precision_at_1 = static_cast<double>(hits_at_1) / static_cast<double>(queries_with_ranks);
                precision_at_5 = static_cast<double>(hits_at_5) / static_cast<double>(queries_with_ranks);
                precision_at_10 = static_cast<double>(hits_at_10) / static_cast<double>(queries_with_ranks);
                
                // For R=1 (single ground truth), Precision@K = Recall@K
                recall_at_1 = precision_at_1;
                recall_at_5 = precision_at_5;
                recall_at_10 = precision_at_10;
                
                std::cout << "[DEBUG] P@K: P@1=" << precision_at_1 << " P@5=" << precision_at_5 
                          << " P@10=" << precision_at_10 << std::endl;
            }
        }

        if (!image_retrieval_ap_per_query.empty()) {
            double retrieval_sum = std::accumulate(
                image_retrieval_ap_per_query.begin(),
                image_retrieval_ap_per_query.end(),
                0.0);
            image_retrieval_map = retrieval_sum / static_cast<double>(image_retrieval_ap_per_query.size());
        } else {
            image_retrieval_map = -1.0;
        }
    }
    
    /**
     * @brief Add precision result for a specific image and scene
     */
    void addImageResult(const std::string& scene_name, double precision, int matches, int keypoints) {
        // Add to overall metrics
        precisions_per_image.push_back(precision);
        total_matches += matches;
        total_keypoints += keypoints;
        total_images_processed++;
        
        // Update per-scene statistics - properly initialize if first time
        if (per_scene_precisions.find(scene_name) == per_scene_precisions.end()) {
            per_scene_precisions[scene_name] = std::vector<double>();
            per_scene_matches[scene_name] = 0;
            per_scene_keypoints[scene_name] = 0;
            per_scene_image_count[scene_name] = 0;
        }
        
        // Store precision for proper averaging later
        per_scene_precisions[scene_name].push_back(precision);
        per_scene_matches[scene_name] += matches;
        per_scene_keypoints[scene_name] += keypoints;
        per_scene_image_count[scene_name]++;
    }

    /**
     * @brief Add Average Precision result for a query (true IR-style mAP)
     * @param scene_name Scene identifier
     * @param ap_result Result from TrueAveragePrecision computation
     */
    void addQueryAP(const std::string& scene_name, const TrueAveragePrecision::QueryAPResult& ap_result) {
        if (ap_result.has_potential_match) {
            ap_per_query.push_back(ap_result.ap);
            per_scene_ap[scene_name].push_back(ap_result.ap);
            ranks_per_query.push_back(ap_result.rank_of_true_match);  // Store rank for P@K/R@K
            total_queries_processed++;
        } else {
            total_queries_excluded++;
            per_scene_excluded[scene_name]++; // Track R=0 queries per scene
            ranks_per_query.push_back(-1);    // R=0 query (no rank)
        }
    }

    /**
     * @brief Add image-retrieval AP for a query image
     * @param scene_name Scene identifier for the query
     * @param ap Average precision for the ranked candidate list
     */
    void addImageRetrievalAP(const std::string& scene_name, double ap) {
        image_retrieval_ap_per_query.push_back(ap);
        image_retrieval_ap_per_scene[scene_name].push_back(ap);
        image_retrieval_queries = static_cast<int>(image_retrieval_ap_per_query.size());
    }

    /**
     * @brief Merge another ExperimentMetrics into this one
     * @param other The metrics to merge in
     */
    void merge(const ExperimentMetrics& other) {
        // Merge success/error status
        if (!other.success) {
            success = false;
            if (!error_message.empty()) {
                error_message += "; ";
            }
            error_message += other.error_message;
        }
        
        // Merge per-image precisions using insert for efficiency
        precisions_per_image.insert(
            precisions_per_image.end(),
            other.precisions_per_image.begin(),
            other.precisions_per_image.end()
        );
        
        // Merge counters
        total_matches += other.total_matches;
        total_keypoints += other.total_keypoints;
        total_images_processed += other.total_images_processed;
        total_queries_processed += other.total_queries_processed;
        total_queries_excluded += other.total_queries_excluded;
        
        // Merge per-scene statistics safely (no throws on missing keys)
        for (const auto& kv : other.per_scene_precisions) {
            const auto& scene = kv.first;
            const auto& precisions = kv.second;

            // Merge precision vectors
            auto& dst_vec = per_scene_precisions[scene];
            dst_vec.insert(dst_vec.end(), precisions.begin(), precisions.end());

            // Safe merge with default 0 if key missing
            per_scene_matches[scene] += (other.per_scene_matches.count(scene) ? 
                                       other.per_scene_matches.at(scene) : 0);
            per_scene_keypoints[scene] += (other.per_scene_keypoints.count(scene) ? 
                                         other.per_scene_keypoints.at(scene) : 0);
            per_scene_image_count[scene] += (other.per_scene_image_count.count(scene) ? 
                                           other.per_scene_image_count.at(scene) : 
                                           static_cast<int>(precisions.size())); // fallback to precision count
        }
        
        // Merge true mAP AP values
        ap_per_query.insert(
            ap_per_query.end(),
            other.ap_per_query.begin(),
            other.ap_per_query.end()
        );
        
        for (const auto& kv : other.per_scene_ap) {
            const auto& scene = kv.first;
            const auto& aps = kv.second;
            
            auto& dst_aps = per_scene_ap[scene];
            dst_aps.insert(dst_aps.end(), aps.begin(), aps.end());
        }
        
        // Merge per-scene excluded counts (R=0 queries)
        for (const auto& kv : other.per_scene_excluded) {
            per_scene_excluded[kv.first] += kv.second;
        }
        
        // Merge ranks per query (critical for P@K/R@K calculation)
        ranks_per_query.insert(
            ranks_per_query.end(),
            other.ranks_per_query.begin(),
            other.ranks_per_query.end()
        );

        image_retrieval_ap_per_query.insert(
            image_retrieval_ap_per_query.end(),
            other.image_retrieval_ap_per_query.begin(),
            other.image_retrieval_ap_per_query.end()
        );

        for (const auto& [scene_name, aps] : other.image_retrieval_ap_per_scene) {
            auto& bucket = image_retrieval_ap_per_scene[scene_name];
            bucket.insert(bucket.end(), aps.begin(), aps.end());
        }

        image_retrieval_queries = static_cast<int>(image_retrieval_ap_per_query.size());
    }

    /**
     * @brief Get average precision for a specific scene
     * @param scene_name The scene to get average precision for
     * @return Average precision for the scene, or 0.0 if scene not found
     */
    double getSceneAveragePrecision(const std::string& scene_name) const {
        auto it = per_scene_precisions.find(scene_name);
        if (it != per_scene_precisions.end() && !it->second.empty()) {
            double sum = 0.0;
            for (double p : it->second) {
                sum += p;
            }
            return sum / it->second.size();
        }
        return 0.0;
    }

    /**
     * @brief Get all scene names with results
     */
    std::vector<std::string> getSceneNames() const {
        std::vector<std::string> names;
        for (const auto& [scene_name, precisions] : per_scene_precisions) {
            if (!precisions.empty()) {
                names.push_back(scene_name);
            }
        }
        return names;
    }

    /**
     * @brief Create error metrics with message
     */
    static ExperimentMetrics createError(const std::string& message) {
        ExperimentMetrics error_metrics;
        error_metrics.success = false;
        error_metrics.error_message = message;
        return error_metrics;
    }

    /**
     * @brief Create successful empty metrics
     */
    static ExperimentMetrics createSuccess() {
        ExperimentMetrics success_metrics;
        success_metrics.success = true;
        return success_metrics;
    }
};

#endif // CORE_METRICS_EXPERIMENT_METRICS_HPP
