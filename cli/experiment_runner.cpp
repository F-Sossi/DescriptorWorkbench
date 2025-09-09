#include "src/core/config/YAMLConfigLoader.hpp"
#include "src/core/descriptor/extractors/wrappers/DNNPatchWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.hpp"
#include "thesis_project/logging.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "src/core/metrics/TrueAveragePrecision.hpp"
#include "thesis_project/types.hpp"
#ifdef BUILD_DATABASE
#include "thesis_project/database/DatabaseManager.hpp"
#endif
#include <iostream>
#include <filesystem>
#include <numeric>
#include <chrono>
#include <fstream>

using namespace thesis_project;

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
#ifdef BUILD_DATABASE
    thesis_project::database::DatabaseManager* db_ptr,
#else
    void* db_ptr,
#endif
    ProfilingSummary& profile
) {
    namespace fs = std::filesystem;
    ::ExperimentMetrics overall;
    overall.success = true;

    try {
        if (!fs::exists(yaml_config.dataset.path) || !fs::is_directory(yaml_config.dataset.path)) {
            return ::ExperimentMetrics::createError("Invalid data folder: " + yaml_config.dataset.path);
        }

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
        // Matching: use brute-force L2 with cross-check (current default)
        auto matcher = thesis_project::matching::MatchingFactory::createStrategy(BRUTE_FORCE);

        // Profiling accumulators
        double detect_ms = 0.0;
        double compute_ms = 0.0;
        double match_ms = 0.0;
        long total_images = 0;
        long total_kps = 0;

        for (const auto& entry : fs::directory_iterator(yaml_config.dataset.path)) {
            if (!entry.is_directory()) continue;
            const std::string scene_folder = entry.path().string();
            const std::string scene_name = entry.path().filename().string();

            // Filter scenes if specified in config
            if (!yaml_config.dataset.scenes.empty()) {
                bool scene_found = false;
                for (const auto& allowed_scene : yaml_config.dataset.scenes) {
                    if (scene_name == allowed_scene) {
                        scene_found = true;
                        break;
                    }
                }
                if (!scene_found) continue;
            }

            ::ExperimentMetrics metrics;

            // Load image1
            std::string image1_path = scene_folder + "/1.ppm";
            cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
            if (image1.empty()) continue;
            if (!desc_config.params.use_color && image1.channels() > 1) {
                cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
            }

            // Get keypoints for image1
            std::vector<cv::KeyPoint> keypoints1;
#ifdef BUILD_DATABASE
            if (yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION && db_ptr) {
                auto& db = *db_ptr;
                keypoints1 = db.getLockedKeypoints(scene_name, "1.ppm");
                if (keypoints1.empty()) {
                    LOG_ERROR("No locked keypoints for " + scene_name + "/1.ppm");
                    continue;
                }
            } else
#endif
            {
                // Detect fresh keypoints
                auto det = makeDetector(yaml_config);
                auto t0 = std::chrono::high_resolution_clock::now();
                det->detect(image1, keypoints1);
                auto t1 = std::chrono::high_resolution_clock::now();
                detect_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                LOG_INFO("Detected " + std::to_string(keypoints1.size()) + " keypoints for " + scene_name + "/1.ppm");
            }

            // Compute descriptors1 via new interface + pooling
            cv::Mat descriptors1;
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                try {
                    descriptors1 = pooling->computeDescriptors(image1, keypoints1, *extractor, desc_config);
                    LOG_INFO("Computed descriptors1: " + std::to_string(descriptors1.rows) + "x" + std::to_string(descriptors1.cols));
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to compute descriptors for " + scene_name + "/1.ppm: " + std::string(e.what()));
                    continue;
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            }

            for (int i = 2; i <= 6; ++i) {
                std::string image_name = std::to_string(i) + ".ppm";
                std::string image2_path = scene_folder + "/" + image_name;
                cv::Mat image2 = cv::imread(image2_path, cv::IMREAD_COLOR);
                if (image2.empty()) continue;
            if (!desc_config.params.use_color && image2.channels() > 1) {
                cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
            }

                // Get keypoints2
                std::vector<cv::KeyPoint> keypoints2;
#ifdef BUILD_DATABASE
                if (yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION && db_ptr) {
                    auto& db = *db_ptr;
                    keypoints2 = db.getLockedKeypoints(scene_name, image_name);
                    if (keypoints2.empty()) {
                        LOG_ERROR("No locked keypoints for " + scene_name + "/" + image_name);
                        continue;
                    }
                } else
#endif
                {
                    auto det = makeDetector(yaml_config);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    det->detect(image2, keypoints2);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    detect_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                }

                // Compute descriptors2
                cv::Mat descriptors2;
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    descriptors2 = pooling->computeDescriptors(image2, keypoints2, *extractor, desc_config);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                }
                if (descriptors1.empty() || descriptors2.empty()) continue;

                // Match descriptors
                std::vector<cv::DMatch> matches;
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    matches = matcher->matchDescriptors(descriptors1, descriptors2);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    match_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                }

                // Legacy precision using index equality (if locked)
                int correctMatches = 0;
                if (yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION && !matches.empty()) {
                    for (const auto& m : matches) if (m.queryIdx == m.trainIdx) ++correctMatches;
                    double precision = matches.empty() ? 0.0 : (double)correctMatches / matches.size();
                    metrics.addImageResult(scene_name, precision, (int)matches.size(), (int)keypoints2.size());
                }

                // True mAP via homography if available
                std::string Hpath = scene_folder + "/H_1_" + std::to_string(i);
                cv::Mat H = cv::Mat();
                std::ifstream hfile(Hpath);
                if (hfile.good()) {
                    H = cv::Mat::zeros(3,3,CV_64F);
                    for (int r=0;r<3;++r) for (int c=0;c<3;++c) hfile >> H.at<double>(r,c);
                    hfile.close();
                }
                if (!H.empty() && !keypoints1.empty() && !keypoints2.empty()) {
                    for (int q = 0; q < (int)keypoints1.size(); ++q) {
                        cv::Mat qdesc = descriptors1.row(q);
                        if (qdesc.empty() || cv::norm(qdesc) == 0.0) {
                            auto dummy = TrueAveragePrecision::QueryAPResult{}; dummy.ap = 0.0; dummy.has_potential_match=false;
                            metrics.addQueryAP(scene_name, dummy);
                            continue;
                        }
                        std::vector<double> dists; dists.reserve(keypoints2.size());
                        for (int t = 0; t < (int)keypoints2.size(); ++t) {
                            cv::Mat tdesc = descriptors2.row(t);
                            if (tdesc.empty()) { dists.push_back(std::numeric_limits<double>::infinity()); continue; }
                            double dist = cv::norm(qdesc, tdesc, cv::NORM_L2SQR);
                            dists.push_back(dist);
                        }
                        auto ap = TrueAveragePrecision::computeQueryAP(
                            keypoints1[q], H, keypoints2, dists, 3.0
                        );
                        metrics.addQueryAP(scene_name, ap);
                    }
                }
            }

            // finalize per-scene
            metrics.calculateMeanPrecision();
            overall.merge(metrics);
            total_images += 5;
            total_kps += static_cast<long>(keypoints1.size());
        }

        overall.calculateMeanPrecision();
        overall.success = true;
        // Export profiling to caller
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

#ifdef BUILD_DATABASE
        // Initialize database for experiment tracking (respect YAML when provided)
        auto normalizeDbPath = [](std::string s) {
            const std::string prefix = "sqlite:///";
            if (s.rfind(prefix, 0) == 0) {
                return s.substr(prefix.size());
            }
            return s;
        };

        bool db_enabled = true;
        std::string db_path = "experiments.db"; // default in current working dir (build/)
        if (!yaml_config.database.connection_string.empty()) {
            db_path = normalizeDbPath(yaml_config.database.connection_string);
        }
        if (!yaml_config.database.connection_string.empty() || yaml_config.database.enabled) {
            db_enabled = true;
        }

        thesis_project::database::DatabaseManager db(db_path, db_enabled);
        if (db.isEnabled()) {
            db.optimizeForBulkOperations();
            LOG_INFO("Database tracking enabled");
        } else {
            LOG_INFO("Database tracking disabled");
        }
#endif

        LOG_INFO("Experiment: " + yaml_config.experiment.name);
        LOG_INFO("Description: " + yaml_config.experiment.description);
        LOG_INFO("Dataset: " + yaml_config.dataset.path);
        LOG_INFO("Descriptors: " + std::to_string(yaml_config.descriptors.size()));

        // Results directory creation removed - using database storage only
        std::string results_base = yaml_config.output.results_path + yaml_config.experiment.name;

        // Run experiment for each descriptor configuration
        for (size_t i = 0; i < yaml_config.descriptors.size(); ++i) {
            const auto& desc_config = yaml_config.descriptors[i];

            LOG_INFO("Running experiment with descriptor: " + desc_config.name);

            // Descriptor-specific directory creation removed - using database storage only
            std::string results_path = results_base + "/" + desc_config.name;

            // New interface is the default path where supported; processor_utils handles routing.

#ifdef BUILD_DATABASE
            // Record experiment configuration
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
            
            auto start_time = std::chrono::high_resolution_clock::now();
            int experiment_id = db.recordConfiguration(dbConfig);
#endif

            // Run new pipeline path end-to-end
            ProfilingSummary profile{};
            auto experiment_metrics = processDirectoryNew(yaml_config, desc_config,
#ifdef BUILD_DATABASE
                &db,
#else
                nullptr,
#endif
                profile);
            
#ifdef BUILD_DATABASE
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
                
                db.recordExperiment(results);
            }
#endif

            if (experiment_metrics.success) {
                LOG_INFO("✅ Completed descriptor: " + desc_config.name);
            } else {
                LOG_ERROR("❌ Failed descriptor: " + desc_config.name);
            }
        }

        LOG_INFO("🎉 Experiment completed: " + yaml_config.experiment.name);
        LOG_INFO("📊 Experiment results saved to database");

        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Experiment failed: " + std::string(e.what()));
        return 1;
    }
}
