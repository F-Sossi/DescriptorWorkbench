#include "src/core/config/YAMLConfigLoader.hpp"
#include "src/core/descriptor/extractors/wrappers/DNNPatchWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.hpp"
#include "thesis_project/logging.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
// #include "thesis_project/keypoints/KeypointAttributeAdapter.hpp" // No longer needed with pure intersection sets
#include "src/core/experiment/ExperimentHelpers.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/pooling/PoolingStrategy.hpp"
#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/matching/MatchingStrategy.hpp"
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
#include <unordered_map>
#include <map>
#ifdef _OPENMP
#include <omp.h>
#include <thread>
#endif

static void printUsage(const std::string& binaryName) {
    std::cout << "Usage: " << binaryName << " <config.yaml>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help        Show this help message and exit" << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << binaryName << " ../config/experiments/sift_baseline.yaml" << std::endl;
}

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

struct ImageRetrievalAccumulator;

// Context for scene processing (all dependencies needed for parallel execution)
struct SceneProcessingContext {
    const config::ExperimentConfig& yaml_config;
    const config::ExperimentConfig::DescriptorConfig& desc_config;
    thesis_project::database::DatabaseManager* db_ptr;
    int experiment_id;
    IDescriptorExtractor& extractor;
    pooling::PoolingStrategy& pooling;
    matching::MatchingStrategy& matcher;
    ImageRetrievalAccumulator* retrieval_accumulator;
    bool use_db_keypoints;
    int keypoint_set_id;
    bool store_descriptors;
    bool store_matches;
    bool store_visualizations;
    const std::string& processing_method;
};

// Thread-local profiling accumulator
struct ThreadLocalProfiling {
    double detect_ms = 0.0;
    double compute_ms = 0.0;
    double match_ms = 0.0;
    long total_images = 0;
    long total_kps = 0;
};

struct ImageRetrievalAccumulator {
    struct ImageFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    struct QueryKey {
        std::string scene;
        std::string image;
    };

    void registerImage(const std::string& scene_name,
                       const std::string& image_name,
                       const std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& descriptors,
                       bool is_query_image) {
        if (descriptors.empty()) {
            return;
        }

        auto& scene_bucket = feature_store_[scene_name];
        auto& features = scene_bucket[image_name];
        features.keypoints = keypoints;
        features.descriptors = descriptors.clone();

        if (is_query_image) {
            queries_.push_back({scene_name, image_name});
        }
    }

    void compute(const thesis_project::config::ExperimentConfig& config,
                 const std::string& scorer,
                 ::ExperimentMetrics& metrics) {
        if (computed_) {
            return;
        }
        computed_ = true;

        if (queries_.empty()) {
            return;
        }

        const bool scorer_uses_correctness = (scorer == "correct_matches");
        const bool can_evaluate_correctness =
            scorer_uses_correctness &&
            config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION;

        const std::size_t query_count = queries_.size();
        LOG_INFO("Computing image-retrieval candidates for " + std::to_string(query_count) + " queries");

        std::vector<double> ap_per_query(query_count, 0.0);
        std::vector<std::string> scene_per_query(query_count);
        const std::size_t total_candidates = totalCandidateCount();

#ifdef _OPENMP
        bool parallel_queries = config.performance.parallel_scenes && query_count > 1;
#else
        bool parallel_queries = false;
#endif

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) if(parallel_queries)
#endif
        for (std::size_t idx = 0; idx < query_count; ++idx) {
            const auto& query = queries_[idx];
            const ImageFeatures* query_features = findFeatures(query.scene, query.image);
            if (!query_features) {
                scene_per_query[idx] = query.scene;
                ap_per_query[idx] = 0.0;
                continue;
            }

            auto matcher = thesis_project::matching::MatchingFactory::createStrategy(
                config.evaluation.params.matching_method);

            struct CandidateScore {
                double score = 0.0;
                bool relevant = false;
                std::string scene;
                std::string image;
            };

            std::vector<CandidateScore> candidates;
            if (total_candidates > 0) {
                candidates.reserve(total_candidates - 1); // exclude self
            }

            for (const auto& [candidate_scene, images] : feature_store_) {
                for (const auto& [candidate_image, candidate_features] : images) {
                    if (candidate_scene == query.scene && candidate_image == query.image) {
                        continue; // skip self
                    }

                    bool evaluate_correctness = can_evaluate_correctness && (candidate_scene == query.scene);

                    auto artifacts = experiment_helpers::computeMatches(
                        query_features->descriptors,
                        candidate_features.descriptors,
                        *matcher,
                        evaluate_correctness,
                        query_features->keypoints,
                        candidate_features.keypoints);

                    if (artifacts.matches.empty()) {
                        CandidateScore cs;
                        cs.score = 0.0;
                        cs.relevant = (candidate_scene == query.scene);
                        cs.scene = candidate_scene;
                        cs.image = candidate_image;
                        candidates.push_back(std::move(cs));
                        continue;
                    }

                    CandidateScore cs;
                    cs.score = computeScore(artifacts, scorer);
                    cs.relevant = (candidate_scene == query.scene);
                    cs.scene = candidate_scene;
                    cs.image = candidate_image;
                    candidates.push_back(std::move(cs));
                }
            }

            std::sort(candidates.begin(), candidates.end(), [](const CandidateScore& a, const CandidateScore& b) {
                if (a.score == b.score) {
                    if (a.scene == b.scene) {
                        return a.image < b.image;
                    }
                    return a.scene < b.scene;
                }
                return a.score > b.score;
            });

            std::vector<bool> relevance;
            relevance.reserve(candidates.size());
            for (const auto& entry : candidates) {
                relevance.push_back(entry.relevant);
            }

            const double ap = computeAveragePrecision(relevance);
            scene_per_query[idx] = query.scene;
            ap_per_query[idx] = ap;
        }

        for (std::size_t idx = 0; idx < query_count; ++idx) {
            metrics.addImageRetrievalAP(scene_per_query[idx], ap_per_query[idx]);
        }
    }

private:
    const ImageFeatures* findFeatures(const std::string& scene, const std::string& image) const {
        auto scene_it = feature_store_.find(scene);
        if (scene_it == feature_store_.end()) {
            return nullptr;
        }
        auto& image_map = scene_it->second;
        auto image_it = image_map.find(image);
        if (image_it == image_map.end()) {
            return nullptr;
        }
        return &image_it->second;
    }

    std::size_t totalCandidateCount() const {
        std::size_t count = 0;
        for (const auto& [scene, images] : feature_store_) {
            count += images.size();
        }
        return count;
    }

    static double computeScore(const experiment_helpers::MatchArtifacts& artifacts,
                               const std::string& scorer) {
        if (scorer == "total_matches") {
            return static_cast<double>(artifacts.matches.size());
        }

        if (scorer == "ratio_sum") {
            double sum = 0.0;
            for (const auto& match : artifacts.matches) {
                sum += 1.0 / (1.0 + match.distance);
            }
            return sum;
        }

        if (scorer == "correct_matches") {
            return static_cast<double>(artifacts.correctMatches);
        }

        // Fallback: total matches when scorer is unrecognized
        return static_cast<double>(artifacts.matches.size());
    }

    static double computeAveragePrecision(const std::vector<bool>& relevance) {
        int relevant_count = 0;
        double accumulated_precision = 0.0;

        for (size_t i = 0; i < relevance.size(); ++i) {
            if (!relevance[i]) {
                continue;
            }
            relevant_count++;
            accumulated_precision += static_cast<double>(relevant_count) / static_cast<double>(i + 1);
        }

        if (relevant_count == 0) {
            return 0.0;
        }

        return accumulated_precision / static_cast<double>(relevant_count);
    }

    std::map<std::string, std::map<std::string, ImageFeatures>> feature_store_;
    std::vector<QueryKey> queries_;
    bool computed_ = false;
};

// Create a simple SIFT detector for independent detection
static cv::Ptr<cv::Feature2D> makeDetector(const thesis_project::config::ExperimentConfig& cfg) {
    // Only SIFT supported here for simplicity; extend as needed
    int maxf = cfg.keypoints.params.max_features;
    if (maxf > 0) return cv::SIFT::create(maxf);
    return cv::SIFT::create();
}

/**
 * @brief Process a single scene (6 images) - THREAD-SAFE for OpenMP parallelization
 *
 * This function processes one scene independently and can be called in parallel.
 * Each thread gets its own detector and profiling accumulator.
 */
static std::pair<::ExperimentMetrics, ThreadLocalProfiling> processSingleScene(
    const std::string& scene_folder,
    const std::string& scene_name,
    const SceneProcessingContext& ctx
) {
    namespace fs = std::filesystem;
    ::ExperimentMetrics metrics;
    ThreadLocalProfiling prof;

    // Thread-local detector (not shared between threads)
    cv::Ptr<cv::Feature2D> detector;
    auto ensureDetector = [&]() -> cv::Ptr<cv::Feature2D> {
        if (!detector) {
            detector = makeDetector(ctx.yaml_config);
        }
        return detector;
    };

    auto detectKeypoints = [&](const cv::Mat& image,
                              const std::string& scene_name,
                              const std::string& image_name,
                              std::vector<cv::KeyPoint>& keypoints) {
        const auto det = ensureDetector();
        auto t0 = std::chrono::high_resolution_clock::now();
        det->detect(image, keypoints);
        auto t1 = std::chrono::high_resolution_clock::now();
        prof.detect_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (image_name == "1.ppm") {
            LOG_INFO("Detected " + std::to_string(keypoints.size()) + " keypoints for " + scene_name + "/" + image_name);
        }
    };

    auto loadKeypointsFromDatabase = [&](const std::string& scene_name,
                                         const std::string& image_name,
                                         std::vector<cv::KeyPoint>& keypoints) -> bool {
        if (!ctx.db_ptr) return false;
        auto& db = *ctx.db_ptr;
        bool loaded = false;
        if (ctx.keypoint_set_id >= 0) {
            keypoints = db.getLockedKeypointsFromSet(ctx.keypoint_set_id, scene_name, image_name);
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
            image, keypoints, ctx.extractor, ctx.pooling, ctx.desc_config);
        if (descriptors.empty()) {
            LOG_ERROR("Failed to compute descriptors for " + log_prefix);
            return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        prof.compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return true;
    };

    auto maybeStoreDescriptors = [&](const std::string& scene_name,
                                     const std::string& image_name,
                                     const std::vector<cv::KeyPoint>& keypoints,
                                     const cv::Mat& descriptors) {
        if (!ctx.store_descriptors || descriptors.empty() || !ctx.db_ptr) return;
        if (!ctx.db_ptr->storeDescriptors(ctx.experiment_id, scene_name, image_name,
                                      keypoints, descriptors, ctx.processing_method)) {
            LOG_WARNING("Failed to store descriptors for " + scene_name + "/" + image_name);
        }
    };

    auto maybeStoreMatches = [&](const std::string& scene_name,
                                 const std::string& image_name,
                                 const std::vector<cv::KeyPoint>& keypoints1,
                                 const std::vector<cv::KeyPoint>& keypoints2,
                                 const experiment_helpers::MatchArtifacts& artifacts) {
        if (!ctx.store_matches || artifacts.matches.empty() || !ctx.db_ptr) return;
        if (!ctx.db_ptr->storeMatches(ctx.experiment_id, scene_name, "1.ppm", image_name,
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
        if (!ctx.store_visualizations || artifacts.matches.empty() || !ctx.db_ptr) return;
        cv::Mat match_viz = generateMatchVisualization(image1, image2, keypoints1, keypoints2,
                                                       artifacts.matches, artifacts.correctnessFlags);
        if (match_viz.empty()) return;

        std::string image_pair = "1_" + std::to_string(image_index);
        double precision = artifacts.matches.empty() ? 0.0 : static_cast<double>(artifacts.correctMatches) / artifacts.matches.size();
        std::string metadata = "{\"matches\":" + std::to_string(artifacts.matches.size()) +
                              ",\"correct\":" + std::to_string(artifacts.correctMatches) +
                              ",\"precision\":" + std::to_string(precision) + "}";

        if (!ctx.db_ptr->storeVisualization(ctx.experiment_id, scene_name, "matches", image_pair, match_viz, metadata)) {
            LOG_WARNING("Failed to store visualization for " + scene_name + "/" + image_pair);
        }
    };

    // Process reference image (1.ppm)
    const std::string base_image_name = "1.ppm";
    const std::string base_image_path = scene_folder + "/" + base_image_name;
    cv::Mat image1 = cv::imread(base_image_path, cv::IMREAD_COLOR);
    if (image1.empty()) return {metrics, prof};
    if (!ctx.desc_config.params.use_color && image1.channels() > 1) {
        cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    }

    std::vector<cv::KeyPoint> keypoints1;
    if (ctx.use_db_keypoints) {
        if (!loadKeypointsFromDatabase(scene_name, base_image_name, keypoints1)) {
            return {metrics, prof};
        }
    } else {
        detectKeypoints(image1, scene_name, base_image_name, keypoints1);
    }

    cv::Mat descriptors1;
    if (!computeDescriptors(image1, keypoints1, descriptors1, scene_name + "/" + base_image_name)) {
        return {metrics, prof};
    }

    maybeStoreDescriptors(scene_name, base_image_name, keypoints1, descriptors1);
    if (ctx.retrieval_accumulator) {
        ctx.retrieval_accumulator->registerImage(
            scene_name,
            base_image_name,
            keypoints1,
            descriptors1,
            true);
    }
    prof.total_images += 1;
    prof.total_kps += static_cast<long>(keypoints1.size());

    // Process images 2-6
    for (int image_index = 2; image_index <= 6; ++image_index) {
        const std::string image_name = std::to_string(image_index) + ".ppm";
        const std::string image_path = scene_folder + "/" + image_name;
        cv::Mat image2 = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image2.empty()) continue;
        if (!ctx.desc_config.params.use_color && image2.channels() > 1) {
            cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
        }

        std::vector<cv::KeyPoint> keypoints2;
        if (ctx.use_db_keypoints) {
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
        if (ctx.retrieval_accumulator) {
            ctx.retrieval_accumulator->registerImage(
                scene_name,
                image_name,
                keypoints2,
                descriptors2,
                false);
        }
        prof.total_images += 1;
        prof.total_kps += static_cast<long>(keypoints2.size());

        if (descriptors1.empty() || descriptors2.empty()) continue;

        auto match_t0 = std::chrono::high_resolution_clock::now();
        bool evaluateCorrectness = ctx.yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION;
        auto artifacts = experiment_helpers::computeMatches(
            descriptors1, descriptors2, ctx.matcher, evaluateCorrectness, keypoints1, keypoints2);
        auto match_t1 = std::chrono::high_resolution_clock::now();
        prof.match_ms += std::chrono::duration_cast<std::chrono::milliseconds>(match_t1 - match_t0).count();

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
    return {metrics, prof};
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

        ImageRetrievalAccumulator retrieval_accumulator;
        ImageRetrievalAccumulator* retrieval_ptr = nullptr;
        if (yaml_config.evaluation.params.image_retrieval.enabled) {
            retrieval_ptr = &retrieval_accumulator;
        }

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

        // Create scene processing context for parallel execution
        SceneProcessingContext ctx{
            yaml_config,
            desc_config,
            db_ptr,
            experiment_id,
            *extractor,
            *pooling,
            *matcher,
            retrieval_ptr,
            use_db_keypoints,
            keypoint_set_id,
            store_descriptors,
            store_matches,
            store_visualizations,
            processing_method
        };

        // Collect all scene paths that match the filter
        std::vector<std::pair<std::string, std::string>> scenes_to_process;
        for (const auto& entry : fs::directory_iterator(yaml_config.dataset.path)) {
            if (!entry.is_directory()) continue;

            const std::string scene_name = entry.path().filename().string();
            if (!yaml_config.dataset.scenes.empty()) {
                bool scene_allowed = std::find(
                    yaml_config.dataset.scenes.begin(),
                    yaml_config.dataset.scenes.end(),
                    scene_name) != yaml_config.dataset.scenes.end();
                if (!scene_allowed) continue;
            }

            scenes_to_process.emplace_back(entry.path().string(), scene_name);
        }

        // Determine thread count for OpenMP
        const bool retrieval_enabled = yaml_config.evaluation.params.image_retrieval.enabled;
        const bool use_parallel = yaml_config.performance.parallel_scenes && !retrieval_enabled;
        int num_threads = yaml_config.performance.num_threads;

#ifdef _OPENMP
        if (retrieval_enabled && yaml_config.performance.parallel_scenes) {
            LOG_INFO("Image retrieval evaluation enabled: processing scenes sequentially for consistency");
        }
        if (use_parallel) {
            if (num_threads <= 0) {
                num_threads = std::thread::hardware_concurrency();
                if (num_threads <= 0) num_threads = 4; // Fallback
            }
            omp_set_num_threads(num_threads);
            LOG_INFO("OpenMP enabled: processing " + std::to_string(scenes_to_process.size()) +
                     " scenes with " + std::to_string(num_threads) + " threads");
        } else {
            LOG_INFO("Sequential processing: parallel_scenes disabled in configuration");
        }
#else
        if (use_parallel) {
            LOG_WARNING("OpenMP not available - running sequentially despite configuration");
        }
#endif

        // Process scenes (parallel or sequential based on configuration)
        std::vector<::ExperimentMetrics> scene_metrics(scenes_to_process.size());
        std::vector<ThreadLocalProfiling> scene_profiling(scenes_to_process.size());

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) if(use_parallel)
#endif
        for (size_t i = 0; i < scenes_to_process.size(); ++i) {
            const auto& [scene_folder, scene_name] = scenes_to_process[i];
            auto result = processSingleScene(scene_folder, scene_name, ctx);
            scene_metrics[i] = result.first;
            scene_profiling[i] = result.second;
        }

        // Merge all scene results
        double detect_ms = 0.0;
        double compute_ms = 0.0;
        double match_ms = 0.0;
        long total_images = 0;
        long total_kps = 0;

        for (size_t i = 0; i < scenes_to_process.size(); ++i) {
            overall.merge(scene_metrics[i]);
            detect_ms += scene_profiling[i].detect_ms;
            compute_ms += scene_profiling[i].compute_ms;
            match_ms += scene_profiling[i].match_ms;
            total_images += scene_profiling[i].total_images;
            total_kps += scene_profiling[i].total_kps;
        }

        if (retrieval_ptr) {
            retrieval_ptr->compute(
                yaml_config,
                yaml_config.evaluation.params.image_retrieval.scorer,
                overall);
        }

        overall.calculateMeanPrecision();
        overall.success = true;

        if (yaml_config.evaluation.params.image_retrieval.enabled) {
            LOG_INFO("Image retrieval MAP: " + std::to_string(overall.image_retrieval_map));
        }

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
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    if (argc != 2) {
        printUsage(argv[0]);
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

                // Record keypoint tracking information
                dbConfig.keypoint_set_id = yaml_config.keypoints.params.keypoint_set_id;
                dbConfig.keypoint_source = toString(yaml_config.keypoints.params.source);

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
                results.image_retrieval_map = experiment_metrics.image_retrieval_map;
                
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
