#include "src/core/config/YAMLConfigLoader.hpp"
#include "src/core/descriptor/extractors/wrappers/DNNPatchWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.hpp"
#include "thesis_project/logging.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
// #include "thesis_project/keypoints/KeypointAttributeAdapter.hpp" // No longer needed with pure intersection sets
#include "src/core/experiment/ExperimentHelpers.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "src/core/metrics/TrueAveragePrecision.hpp"
#include "thesis_project/types.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <limits>
#include <chrono>
#include <fstream>

using namespace thesis_project;
namespace experiment_helpers = thesis_project::experiment;

// Generate match visualization with correctness color coding
static cv::Mat generateMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                                         const std::vector<cv::KeyPoint>& kp1,
                                         const std::vector<cv::KeyPoint>& kp2,
                                         const std::vector<cv::DMatch>& matches,
                                         const std::vector<bool>& correctness) {
    cv::Mat visualization;

    // Ensure both images are color
    cv::Mat color_img1, color_img2;
    if (img1.channels() == 1) {
        cv::cvtColor(img1, color_img1, cv::COLOR_GRAY2BGR);
    } else {
        color_img1 = img1.clone();
    }

    if (img2.channels() == 1) {
        cv::cvtColor(img2, color_img2, cv::COLOR_GRAY2BGR);
    } else {
        color_img2 = img2.clone();
    }

    // Create visualization with color-coded matches
    std::vector<cv::Scalar> match_colors;
    match_colors.reserve(matches.size());

    for (size_t i = 0; i < matches.size(); ++i) {
        if (i < correctness.size() && correctness[i]) {
            match_colors.emplace_back(0, 255, 0); // Green for correct matches
        } else {
            match_colors.emplace_back(0, 0, 255); // Red for incorrect matches
        }
    }

    // Draw matches with custom colors
    cv::drawMatches(color_img1, kp1, color_img2, kp2, matches, visualization,
                   cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                   cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Overlay match lines with correctness colors
    if (!visualization.empty()) {
        int img1_width = color_img1.cols;
        for (size_t i = 0; i < matches.size() && i < match_colors.size(); ++i) {
            const auto& match = matches[i];
            if (match.queryIdx < static_cast<int>(kp1.size()) &&
                match.trainIdx < static_cast<int>(kp2.size())) {

                cv::Point2f pt1 = kp1[match.queryIdx].pt;
                cv::Point2f pt2 = kp2[match.trainIdx].pt;
                pt2.x += img1_width; // Offset for side-by-side layout

                cv::line(visualization, pt1, pt2, match_colors[i], 2);
            }
        }
    }

    return visualization;
}

static void maybeAccumulateTrueAveragePrecisionFromFile(
    const std::string& homographyPath,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& descriptors1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& descriptors2,
    const std::string& sceneName,
    ::ExperimentMetrics& metrics) {

    std::ifstream hfile(homographyPath);
    if (!hfile.good()) {
        return;
    }

    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            hfile >> H.at<double>(r, c);
        }
    }

    if (H.empty()) {
        return;
    }

    experiment_helpers::accumulateTrueAveragePrecision(
        H, keypoints1, descriptors1, keypoints2, descriptors2, sceneName, metrics);
}

struct ProfilingSummary {
    double detect_ms = 0.0;
    double compute_ms = 0.0;
    double match_ms = 0.0;
    long total_images = 0;
    long total_kps = 0;
};
// Create a simple SIFT detector for independent detection
static cv::Ptr<cv::Feature2D> makeDetector(const thesis_project::config::ExperimentConfig& cfg) {
    // Only SIFT supported here for simplicity; extend as needed
    int maxf = cfg.keypoints.params.max_features;
    if (maxf > 0) return cv::SIFT::create(maxf);
    return cv::SIFT::create();
}

static ::ExperimentMetrics processDirectoryNew(
    const config::ExperimentConfig& yaml_config,
    const config::ExperimentConfig::DescriptorConfig& desc_config,
    thesis_project::database::DatabaseManager* db_ptr,
    int experiment_id,
    ProfilingSummary& profile
) {
    namespace fs = std::filesystem;
    ::ExperimentMetrics overall;
    overall.success = true;

    try {
        if (!fs::exists(yaml_config.dataset.path) || !fs::is_directory(yaml_config.dataset.path)) {
            return ::ExperimentMetrics::createError("Invalid data folder: " + yaml_config.dataset.path);
        }

        const int keypoint_set_id = yaml_config.keypoints.params.keypoint_set_id;
        const bool use_db_keypoints = db_ptr &&
            (yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION ||
             yaml_config.keypoints.params.source == thesis_project::KeypointSource::INDEPENDENT_DETECTION);

        // Build extractor and pooling strategy for this descriptor (Schema v1)
        std::unique_ptr<IDescriptorExtractor> extractor;
        if (desc_config.type == thesis_project::DescriptorType::DNN_PATCH) {
            if (desc_config.params.dnn_model_path.empty()) {
                throw std::runtime_error("dnn_patch requires dnn.model path in YAML");
            }
            try {
                LOG_INFO("Creating DNNPatchWrapper with model: " + desc_config.params.dnn_model_path);
                extractor = std::make_unique<thesis_project::wrappers::DNNPatchWrapper>(
                    desc_config.params.dnn_model_path,
                    desc_config.params.dnn_input_size,
                    desc_config.params.dnn_support_multiplier,
                    desc_config.params.dnn_rotate_upright,
                    desc_config.params.dnn_mean,
                    desc_config.params.dnn_std,
                    desc_config.params.dnn_per_patch_standardize
                );
                LOG_INFO("DNNPatchWrapper created successfully");
            } catch (const std::exception& e) {
                LOG_WARNING("DNNPatchWrapper failed: " + std::string(e.what()));
                LOG_INFO("Falling back to Lightweight CNN baseline for comparison");
                extractor = std::make_unique<thesis_project::wrappers::PseudoDNNWrapper>(
                    desc_config.params.dnn_input_size,
                    desc_config.params.dnn_support_multiplier,
                    desc_config.params.dnn_rotate_upright
                );
                LOG_INFO("Lightweight CNN baseline created successfully");
            }
        } else {
            extractor = thesis_project::factories::DescriptorFactory::create(desc_config.type);
        }
        auto pooling = thesis_project::pooling::PoolingFactory::createFromConfig(desc_config);
        auto matcher = thesis_project::matching::MatchingFactory::createStrategy(
            yaml_config.evaluation.params.matching_method);

        const bool store_descriptors = db_ptr && db_ptr->isEnabled() && experiment_id != -1 &&
            yaml_config.database.save_descriptors;
        const bool store_matches = db_ptr && db_ptr->isEnabled() && experiment_id != -1 &&
            yaml_config.database.save_matches;
        const bool store_visualizations = db_ptr && db_ptr->isEnabled() && experiment_id != -1 &&
            yaml_config.database.save_visualizations;

        std::string processing_method;
        if (store_descriptors) {
            processing_method = desc_config.name + "-" +
                                toString(desc_config.params.pooling) + "-" +
                                std::to_string(desc_config.params.norm_type);
        }

        // Profiling accumulators
        double detect_ms = 0.0;
        double compute_ms = 0.0;
        double match_ms = 0.0;
        long total_images = 0;
        long total_kps = 0;

        cv::Ptr<cv::Feature2D> detector;
        auto ensureDetector = [&]() -> cv::Ptr<cv::Feature2D> {
            if (!detector) {
                detector = makeDetector(yaml_config);
            }
            return detector;
        };

        auto detectKeypoints = [&](const cv::Mat& image,
                                   const std::string& scene_name,
                                   const std::string& image_name,
                                   std::vector<cv::KeyPoint>& keypoints) {
            auto det = ensureDetector();
            auto t0 = std::chrono::high_resolution_clock::now();
            det->detect(image, keypoints);
            auto t1 = std::chrono::high_resolution_clock::now();
            detect_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            if (image_name == "1.ppm") {
                LOG_INFO("Detected " + std::to_string(keypoints.size()) + " keypoints for " + scene_name + "/" + image_name);
            }
        };

        auto loadKeypointsFromDatabase = [&](const std::string& scene_name,
                                             const std::string& image_name,
                                             std::vector<cv::KeyPoint>& keypoints) -> bool {
            auto& db = *db_ptr;
            bool loaded = false;
            if (keypoint_set_id >= 0) {
                keypoints = db.getLockedKeypointsFromSet(keypoint_set_id, scene_name, image_name);
                loaded = !keypoints.empty();
            } else {
                keypoints = db.getLockedKeypoints(scene_name, image_name);
                loaded = !keypoints.empty();
                LOG_INFO("Experiment not using specified keypoint set");
            }

            if (!loaded) {
                LOG_ERROR("No locked keypoints for " + scene_name + "/" + image_name);
                return false;
            }
            return true;
        };

        auto computeDescriptors = [&](const cv::Mat& image,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      cv::Mat& descriptors,
                                      const std::string& log_prefix) -> bool {
            auto t0 = std::chrono::high_resolution_clock::now();
            descriptors = experiment_helpers::computeDescriptorsWithPooling(
                image, keypoints, *extractor, *pooling, desc_config);
            if (descriptors.empty()) {
                LOG_ERROR("Failed to compute descriptors for " + log_prefix);
                return false;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            return true;
        };

        auto maybeStoreDescriptors = [&](const std::string& scene_name,
                                         const std::string& image_name,
                                         const std::vector<cv::KeyPoint>& keypoints,
                                         const cv::Mat& descriptors) {
            if (!store_descriptors || descriptors.empty()) return;
            if (!db_ptr->storeDescriptors(experiment_id, scene_name, image_name,
                                          keypoints, descriptors, processing_method)) {
                LOG_WARNING("Failed to store descriptors for " + scene_name + "/" + image_name);
            }
        };

        auto maybeStoreMatches = [&](const std::string& scene_name,
                                     const std::string& image_name,
                                     const std::vector<cv::KeyPoint>& keypoints1,
                                     const std::vector<cv::KeyPoint>& keypoints2,
                                     const experiment_helpers::MatchArtifacts& artifacts) {
            if (!store_matches || artifacts.matches.empty()) return;
            if (!db_ptr->storeMatches(experiment_id, scene_name, "1.ppm", image_name,
                                      keypoints1, keypoints2, artifacts.matches, artifacts.correctnessFlags)) {
                LOG_WARNING("Failed to store matches for " + scene_name + "/" + image_name);
            }
        };

        auto maybeStoreVisualization = [&](const std::string& scene_name,
                                           const std::string& image_name,
                                           int image_index,
                                           const cv::Mat& image1,
                                           const cv::Mat& image2,
                                           const std::vector<cv::KeyPoint>& keypoints1,
                                           const std::vector<cv::KeyPoint>& keypoints2,
                                           const experiment_helpers::MatchArtifacts& artifacts) {
            if (!store_visualizations || artifacts.matches.empty()) return;
            cv::Mat match_viz = generateMatchVisualization(image1, image2, keypoints1, keypoints2,
                                                           artifacts.matches, artifacts.correctnessFlags);
            if (match_viz.empty()) return;

            std::string image_pair = "1_" + std::to_string(image_index);
            double precision = artifacts.matches.empty() ? 0.0 : static_cast<double>(artifacts.correctMatches) / artifacts.matches.size();
            std::string metadata = "{\"matches\":" + std::to_string(artifacts.matches.size()) +
                                  ",\"correct\":" + std::to_string(artifacts.correctMatches) +
                                  ",\"precision\":" + std::to_string(precision) + "}";

            if (!db_ptr->storeVisualization(experiment_id, scene_name, "matches", image_pair, match_viz, metadata)) {
                LOG_WARNING("Failed to store visualization for " + scene_name + "/" + image_pair);
            }
        };

        for (const auto& entry : fs::directory_iterator(yaml_config.dataset.path)) {
            if (!entry.is_directory()) continue;

            const std::string scene_folder = entry.path().string();
            const std::string scene_name = entry.path().filename().string();

            if (!yaml_config.dataset.scenes.empty()) {
                bool scene_allowed = std::find(
                    yaml_config.dataset.scenes.begin(),
                    yaml_config.dataset.scenes.end(),
                    scene_name) != yaml_config.dataset.scenes.end();
                if (!scene_allowed) continue;
            }

            ::ExperimentMetrics metrics;

            const std::string base_image_name = "1.ppm";
            const std::string base_image_path = scene_folder + "/" + base_image_name;
            cv::Mat image1 = cv::imread(base_image_path, cv::IMREAD_COLOR);
            if (image1.empty()) continue;
            if (!desc_config.params.use_color && image1.channels() > 1) {
                cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
            }

            std::vector<cv::KeyPoint> keypoints1;
            if (use_db_keypoints) {
                if (!loadKeypointsFromDatabase(scene_name, base_image_name, keypoints1)) {
                    continue;
                }
            } else {
                detectKeypoints(image1, scene_name, base_image_name, keypoints1);
            }

            cv::Mat descriptors1;
            if (!computeDescriptors(image1, keypoints1, descriptors1, scene_name + "/" + base_image_name)) {
                continue;
            }

            maybeStoreDescriptors(scene_name, base_image_name, keypoints1, descriptors1);
            total_images += 1;
            total_kps += static_cast<long>(keypoints1.size());

            for (int image_index = 2; image_index <= 6; ++image_index) {
                const std::string image_name = std::to_string(image_index) + ".ppm";
                const std::string image_path = scene_folder + "/" + image_name;
                cv::Mat image2 = cv::imread(image_path, cv::IMREAD_COLOR);
                if (image2.empty()) continue;
                if (!desc_config.params.use_color && image2.channels() > 1) {
                    cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
                }

                std::vector<cv::KeyPoint> keypoints2;
                if (use_db_keypoints) {
                    if (!loadKeypointsFromDatabase(scene_name, image_name, keypoints2)) {
                        continue;
                    }
                } else {
                    detectKeypoints(image2, scene_name, image_name, keypoints2);
                }

                cv::Mat descriptors2;
                if (!computeDescriptors(image2, keypoints2, descriptors2, scene_name + "/" + image_name)) {
                    continue;
                }

                maybeStoreDescriptors(scene_name, image_name, keypoints2, descriptors2);
                total_images += 1;
                total_kps += static_cast<long>(keypoints2.size());

                if (descriptors1.empty() || descriptors2.empty()) continue;

                auto match_t0 = std::chrono::high_resolution_clock::now();
                bool evaluateCorrectness = yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION;
                auto artifacts = experiment_helpers::computeMatches(
                    descriptors1, descriptors2, *matcher, evaluateCorrectness, keypoints1, keypoints2);
                auto match_t1 = std::chrono::high_resolution_clock::now();
                match_ms += std::chrono::duration_cast<std::chrono::milliseconds>(match_t1 - match_t0).count();

                if (evaluateCorrectness && !artifacts.matches.empty()) {
                    double precision = static_cast<double>(artifacts.correctMatches) / artifacts.matches.size();
                    metrics.addImageResult(scene_name, precision, static_cast<int>(artifacts.matches.size()), static_cast<int>(keypoints2.size()));
                }

                maybeStoreMatches(scene_name, image_name, keypoints1, keypoints2, artifacts);
                maybeStoreVisualization(scene_name, image_name, image_index, image1, image2,
                                         keypoints1, keypoints2, artifacts);

                std::string homography_path = scene_folder + "/H_1_" + std::to_string(image_index);
                maybeAccumulateTrueAveragePrecisionFromFile(
                    homography_path,
                    keypoints1,
                    descriptors1,
                    keypoints2,
                    descriptors2,
                    scene_name,
                    metrics);
            }

            metrics.calculateMeanPrecision();
            overall.merge(metrics);
        }

        overall.calculateMeanPrecision();
        overall.success = true;

        profile.detect_ms = detect_ms;
        profile.compute_ms = compute_ms;
        profile.match_ms = match_ms;
        profile.total_images = total_images;
        profile.total_kps = total_kps;
        return overall;
    } catch (const std::exception& e) {
        return ::ExperimentMetrics::createError(e.what());
    }
}

/**
 * @brief New experiment runner using YAML configuration
 *
 * This CLI tool demonstrates the new configuration system
 * while using existing image processing code.
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        std::cout << "Example: " << argv[0] << " config/experiments/sift_baseline.yaml" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];

    try {
        // Load YAML configuration
        LOG_INFO("Loading experiment configuration from: " + config_path);
        auto yaml_config = config::YAMLConfigLoader::loadFromFile(config_path);

        // Initialize database for experiment tracking (respect YAML when provided)
        auto normalizeDbPath = [](std::string s) {
            const std::string prefix = "sqlite:///";
            if (s.rfind(prefix, 0) == 0) {
                return s.substr(prefix.size());
            }
            return s;
        };

        std::string db_path = yaml_config.database.connection_string.empty()
                                  ? std::string("experiments.db")
                                  : normalizeDbPath(yaml_config.database.connection_string);

        thesis_project::database::DatabaseManager db(db_path, true);
        if (db.isEnabled()) {
            if (db.optimizeForBulkOperations()) {
                LOG_INFO("Bulk operations enabled");
            } else {
                LOG_INFO("Bulk operations disabled");
            }
            LOG_INFO("Database tracking enabled");
        } else {
            LOG_INFO("Database tracking disabled");
        }

        // Resolve keypoint set name to ID for any source type that uses database keypoints
        if (db.isEnabled() && !yaml_config.keypoints.params.keypoint_set_name.empty()) {
            int resolved_set_id = db.getKeypointSetId(yaml_config.keypoints.params.keypoint_set_name);
            if (resolved_set_id == -1) {
                LOG_ERROR("Keypoint set '" + yaml_config.keypoints.params.keypoint_set_name + "' not found in database");
                return 1;
            }
            yaml_config.keypoints.params.keypoint_set_id = resolved_set_id;
            LOG_INFO("Using keypoint set '" + yaml_config.keypoints.params.keypoint_set_name + "' (id=" + std::to_string(resolved_set_id) + ") with source: " + toString(yaml_config.keypoints.params.source));
        }

        LOG_INFO("Experiment: " + yaml_config.experiment.name);
        LOG_INFO("Description: " + yaml_config.experiment.description);
        LOG_INFO("Dataset: " + yaml_config.dataset.path);
        LOG_INFO("Descriptors: " + std::to_string(yaml_config.descriptors.size()));

        // Run experiment for each descriptor configuration
        for (size_t i = 0; i < yaml_config.descriptors.size(); ++i) {
            const auto& desc_config = yaml_config.descriptors[i];

            LOG_INFO("Running experiment with descriptor: " + desc_config.name);

            auto start_time = std::chrono::high_resolution_clock::now();
            int experiment_id = -1;
            if (db.isEnabled()) {
                thesis_project::database::ExperimentConfig dbConfig;
                dbConfig.descriptor_type = desc_config.name;
                dbConfig.dataset_path = yaml_config.dataset.path;
                dbConfig.pooling_strategy = toString(desc_config.params.pooling);
                dbConfig.similarity_threshold = yaml_config.evaluation.params.match_threshold;
                dbConfig.max_features = yaml_config.keypoints.params.max_features;
                dbConfig.parameters["experiment_name"] = yaml_config.experiment.name;
                dbConfig.parameters["descriptor_type"] = toString(desc_config.type);
                dbConfig.parameters["pooling_strategy"] = toString(desc_config.params.pooling);
                dbConfig.parameters["norm_type"] = std::to_string(desc_config.params.norm_type);

                start_time = std::chrono::high_resolution_clock::now();
                experiment_id = db.recordConfiguration(dbConfig);
            }

            // Run new pipeline path end-to-end
            ProfilingSummary profile{};
            auto experiment_metrics = processDirectoryNew(
                yaml_config,
                desc_config,
                db.isEnabled() ? &db : nullptr,
                experiment_id,
                profile);
            
            if (experiment_id != -1) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                
                // Record experiment results
                thesis_project::database::ExperimentResults results;
                results.experiment_id = experiment_id;
                results.descriptor_type = desc_config.name;
                results.dataset_name = yaml_config.dataset.path;
                results.processing_time_ms = duration.count();
                
                // PRIMARY IR-style mAP metrics (NEW: stored as first-class columns)
                results.true_map_macro = experiment_metrics.true_map_macro_by_scene;
                results.true_map_micro = experiment_metrics.true_map_micro;
                results.true_map_macro_with_zeros = experiment_metrics.true_map_macro_by_scene_including_zeros;
                results.true_map_micro_with_zeros = experiment_metrics.true_map_micro_including_zeros;
                
                // Primary MAP metric: prefer macro (scene-balanced) when available, fallback to micro
                results.mean_average_precision = experiment_metrics.true_map_macro_by_scene > 0.0 ? 
                                                experiment_metrics.true_map_macro_by_scene :
                                                experiment_metrics.true_map_micro;
                
                // Legacy precision for backward compatibility (stored in legacy_mean_precision column)
                results.legacy_mean_precision = experiment_metrics.mean_precision;
                
                // Standard metrics
                results.precision_at_1 = experiment_metrics.precision_at_1;
                results.precision_at_5 = experiment_metrics.precision_at_5;
                results.recall_at_1 = experiment_metrics.recall_at_1;
                results.recall_at_5 = experiment_metrics.recall_at_5;
                results.total_matches = experiment_metrics.total_matches;
                results.total_keypoints = experiment_metrics.total_keypoints;
                results.metadata["success"] = experiment_metrics.success ? "true" : "false";
                results.metadata["experiment_name"] = yaml_config.experiment.name;
                // Profiling metadata
                results.metadata["detect_time_ms"] = std::to_string(profile.detect_ms);
                results.metadata["compute_time_ms"] = std::to_string(profile.compute_ms);
                results.metadata["match_time_ms"] = std::to_string(profile.match_ms);
                results.metadata["total_images"] = std::to_string(profile.total_images);
                results.metadata["total_keypoints"] = std::to_string(profile.total_kps);
                double total_sec = duration.count() > 0 ? (duration.count() / 1000.0) : 0.0;
                if (total_sec > 0.0) {
                    results.metadata["images_per_sec"] = std::to_string(profile.total_images / total_sec);
                    results.metadata["kps_per_sec"] = std::to_string(profile.total_kps / total_sec);
                }
                // NOTE: True IR-style mAP metrics are now stored as primary columns, not metadata
                // Query statistics
                results.metadata["total_queries_processed"] = std::to_string(experiment_metrics.total_queries_processed);
                results.metadata["total_queries_excluded"] = std::to_string(experiment_metrics.total_queries_excluded);
                // Precision@K and Recall@K metrics
                results.metadata["precision_at_1"] = std::to_string(experiment_metrics.precision_at_1);
                results.metadata["precision_at_5"] = std::to_string(experiment_metrics.precision_at_5);
                results.metadata["precision_at_10"] = std::to_string(experiment_metrics.precision_at_10);
                results.metadata["recall_at_1"] = std::to_string(experiment_metrics.recall_at_1);
                results.metadata["recall_at_5"] = std::to_string(experiment_metrics.recall_at_5);
                results.metadata["recall_at_10"] = std::to_string(experiment_metrics.recall_at_10);
                // R=0 rate for transparency
                int total_all = experiment_metrics.total_queries_processed + experiment_metrics.total_queries_excluded;
                double r0_rate = total_all > 0 ? (double)experiment_metrics.total_queries_excluded / total_all : 0.0;
                results.metadata["r0_rate"] = std::to_string(r0_rate);
                
                // Per-scene True mAP breakdown
                for (const auto& [scene_name, scene_aps] : experiment_metrics.per_scene_ap) {
                    if (scene_aps.empty()) continue;
                    
                    double scene_ap_sum = std::accumulate(scene_aps.begin(), scene_aps.end(), 0.0);
                    double scene_true_map = scene_ap_sum / static_cast<double>(scene_aps.size());
                    results.metadata[scene_name + "_true_map"] = std::to_string(scene_true_map);
                    results.metadata[scene_name + "_query_count"] = std::to_string(scene_aps.size());
                    
                    // Per-scene with zeros (punitive)
                    int excluded_count = experiment_metrics.per_scene_excluded.count(scene_name) ? 
                                       experiment_metrics.per_scene_excluded.at(scene_name) : 0;
                    int total_scene_queries = static_cast<int>(scene_aps.size()) + excluded_count;
                    if (total_scene_queries > 0) {
                        double scene_true_map_with_zeros = scene_ap_sum / static_cast<double>(total_scene_queries);
                        results.metadata[scene_name + "_true_map_with_zeros"] = std::to_string(scene_true_map_with_zeros);
                        results.metadata[scene_name + "_excluded_count"] = std::to_string(excluded_count);
                    }
                }
                
                if (db.recordExperiment(results)) {
                    LOG_INFO("Results recorded");
                }else {
                    LOG_ERROR("Failed to record results");
                }
            }

            if (experiment_metrics.success) {
                LOG_INFO("Completed descriptor: " + desc_config.name);
            } else {
                LOG_ERROR("❌ Failed descriptor: " + desc_config.name);
            }
        }

        LOG_INFO("Experiment completed: " + yaml_config.experiment.name);
        LOG_INFO("Experiment results saved to database");

        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Experiment failed: " + std::string(e.what()));
        return 1;
    }
}
