#include "result_recording.hpp"
#include "thesis_project/logging.hpp"
#include <numeric>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace thesis_project::cli::experiment_runner_results {

void recordResults(
    thesis_project::database::DatabaseManager& db,
    int experiment_id,
    const std::string& descriptor_name,
    const std::string& dataset_path,
    const ::ExperimentMetrics& metrics,
    const ProfilingSummary& profile,
    const std::string& execution_device,
    const std::string& experiment_name) {

    thesis_project::database::ExperimentResults results;
    results.experiment_id = experiment_id;
    results.descriptor_type = descriptor_name;
    results.dataset_name = dataset_path;
    results.processing_time_ms = 0; // Caller can override if needed

    results.true_map_macro = metrics.true_map_macro_by_scene;
    results.true_map_micro = metrics.true_map_micro;
    results.true_map_macro_with_zeros = metrics.true_map_macro_by_scene_including_zeros;
    results.true_map_micro_with_zeros = metrics.true_map_micro_including_zeros;
    results.image_retrieval_map = metrics.image_retrieval_map;

    results.viewpoint_map = metrics.viewpoint_map;
    results.illumination_map = metrics.illumination_map;
    results.viewpoint_map_with_zeros = metrics.viewpoint_map_including_zeros;
    results.illumination_map_with_zeros = metrics.illumination_map_including_zeros;

    results.keypoint_verification_ap = metrics.keypoint_verification_ap;
    results.verification_viewpoint_ap = metrics.verification_viewpoint_ap;
    results.verification_illumination_ap = metrics.verification_illumination_ap;

    results.keypoint_retrieval_ap = metrics.keypoint_retrieval_ap;
    results.retrieval_viewpoint_ap = metrics.retrieval_viewpoint_ap;
    results.retrieval_illumination_ap = metrics.retrieval_illumination_ap;
    results.retrieval_num_true_positives = metrics.retrieval_num_true_positives;
    results.retrieval_num_hard_negatives = metrics.retrieval_num_hard_negatives;
    results.retrieval_num_distractors = metrics.retrieval_num_distractors;

    results.mean_average_precision = metrics.true_map_macro_by_scene > 0.0
        ? metrics.true_map_macro_by_scene
        : metrics.true_map_micro;

    results.legacy_mean_precision = metrics.mean_precision;

    results.precision_at_1 = metrics.precision_at_1;
    results.precision_at_5 = metrics.precision_at_5;
    results.recall_at_1 = metrics.recall_at_1;
    results.recall_at_5 = metrics.recall_at_5;
    results.total_matches = metrics.total_matches;
    results.total_keypoints = metrics.total_keypoints;
    results.descriptor_time_cpu_ms = profile.compute_ms;
    results.descriptor_time_gpu_ms = 0.0;
    results.match_time_cpu_ms = profile.match_ms;
    results.match_time_gpu_ms = 0.0;
    results.total_pipeline_cpu_ms = profile.detect_ms + profile.compute_ms + profile.match_ms;
    results.total_pipeline_gpu_ms = 0.0;
    results.metadata["success"] = metrics.success ? "true" : "false";

    if (!metrics.success && !metrics.error_message.empty()) {
        LOG_ERROR("Experiment failed for descriptor '" + descriptor_name + "': " + metrics.error_message);
        results.metadata["error_message"] = metrics.error_message;
        std::cerr << "\n[ERROR] Descriptor '" << descriptor_name << "' failed:\n"
                  << metrics.error_message << "\n" << std::endl;
    }

    results.metadata["experiment_name"] = experiment_name;
    results.metadata["execution_device"] = execution_device;
    results.metadata["descriptor_dimension"] = std::to_string(profile.descriptor_dimension);
    results.metadata["detect_time_ms"] = std::to_string(profile.detect_ms);
    results.metadata["compute_time_ms"] = std::to_string(profile.compute_ms);
    results.metadata["match_time_ms"] = std::to_string(profile.match_ms);
    results.metadata["total_images"] = std::to_string(profile.total_images);
    results.metadata["total_keypoints"] = std::to_string(profile.total_kps);

    results.metadata["total_queries_processed"] = std::to_string(metrics.total_queries_processed);
    results.metadata["total_queries_excluded"] = std::to_string(metrics.total_queries_excluded);
    results.metadata["precision_at_1"] = std::to_string(metrics.precision_at_1);
    results.metadata["precision_at_5"] = std::to_string(metrics.precision_at_5);
    results.metadata["precision_at_10"] = std::to_string(metrics.precision_at_10);
    results.metadata["recall_at_1"] = std::to_string(metrics.recall_at_1);
    results.metadata["recall_at_5"] = std::to_string(metrics.recall_at_5);
    results.metadata["recall_at_10"] = std::to_string(metrics.recall_at_10);
    int total_all = metrics.total_queries_processed + metrics.total_queries_excluded;
    double r0_rate = total_all > 0 ? static_cast<double>(metrics.total_queries_excluded) / total_all : 0.0;
    results.metadata["r0_rate"] = std::to_string(r0_rate);

    for (const auto& [scene_name, scene_aps] : metrics.per_scene_ap) {
        if (scene_aps.empty()) continue;

        double scene_ap_sum = std::accumulate(scene_aps.begin(), scene_aps.end(), 0.0);
        double scene_true_map = scene_ap_sum / static_cast<double>(scene_aps.size());
        results.metadata[scene_name + "_true_map"] = std::to_string(scene_true_map);
        results.metadata[scene_name + "_query_count"] = std::to_string(scene_aps.size());

        int excluded_count = metrics.per_scene_excluded.count(scene_name)
            ? metrics.per_scene_excluded.at(scene_name)
            : 0;
        int total_scene_queries = static_cast<int>(scene_aps.size()) + excluded_count;
        if (total_scene_queries > 0) {
            double scene_true_map_with_zeros = scene_ap_sum / static_cast<double>(total_scene_queries);
            results.metadata[scene_name + "_true_map_with_zeros"] = std::to_string(scene_true_map_with_zeros);
            results.metadata[scene_name + "_excluded_count"] = std::to_string(excluded_count);
        }
    }

    if (db.recordExperiment(results)) {
        LOG_INFO("Results recorded");
    } else {
        LOG_ERROR("Failed to record results");
    }

    auto format_ms = [](double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << value;
        return oss.str();
    };
    auto format_score = [](double value) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << value;
        return oss.str();
    };
    const double map_per_ms = profile.compute_ms > 0.0
        ? results.mean_average_precision / profile.compute_ms
        : 0.0;
    LOG_INFO("Efficiency summary -> dim=" + std::to_string(profile.descriptor_dimension) +
             ", compute_ms=" + format_ms(profile.compute_ms) +
             ", match_ms=" + format_ms(profile.match_ms) +
             ", total_cpu_ms=" + format_ms(results.total_pipeline_cpu_ms) +
             ", MAP=" + format_score(results.mean_average_precision) +
             ", MAP/ms=" + format_score(map_per_ms));

    const double hp_delta = results.viewpoint_map - results.illumination_map;
    LOG_INFO("Category breakdown -> HP-V=" + format_score(results.viewpoint_map) +
             ", HP-I=" + format_score(results.illumination_map) +
             ", Delta=" + format_score(hp_delta) +
             " (viewpoint advantage)");
}

} // namespace thesis_project::cli::experiment_runner_results
