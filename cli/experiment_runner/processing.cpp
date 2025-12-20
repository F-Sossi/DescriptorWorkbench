#include "processing.hpp"
#include "cli/experiment_runner/helpers.hpp"
#include "cli/experiment_runner/types.hpp"
#include "src/core/descriptor/extractors/CompositeDescriptorExtractor.hpp"
#include "src/core/descriptor/extractors/wrappers/DNNPatchWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "src/core/experiment/ExperimentHelpers.hpp"
#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/metrics/KeypointVerification.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include "thesis_project/logging.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace thesis_project::cli::experiment_runner_processing {

using namespace thesis_project;
namespace experiment_helpers = thesis_project::experiment;
namespace cli_helpers = thesis_project::cli::experiment_runner_helpers;

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
        features.descriptors = descriptors;

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
                candidates.reserve(total_candidates - 1);
            }

            for (const auto& [candidate_scene, images] : feature_store_) {
                for (const auto& [candidate_image, candidate_features] : images) {
                    if (candidate_scene == query.scene && candidate_image == query.image) {
                        continue;
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

        if (!ap_per_query.empty()) {
            double retrieval_sum = std::accumulate(ap_per_query.begin(), ap_per_query.end(), 0.0);
            metrics.image_retrieval_map = retrieval_sum / static_cast<double>(ap_per_query.size());
        }

        feature_store_.clear();
        queries_.clear();
    }

private:
    [[nodiscard]] const ImageFeatures* findFeatures(const std::string& scene, const std::string& image) const {
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

    [[nodiscard]] std::size_t totalCandidateCount() const {
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

struct VerificationAccumulator {
    struct SceneData {
        std::string scene_name;
        std::map<std::string, std::vector<cv::KeyPoint>> keypoints_per_image;
        std::map<std::string, cv::Mat> descriptors_per_image;
        std::map<std::string, cv::Mat> homographies;
    };

    void registerImage(const std::string& scene_name,
                       const std::string& image_name,
                       const std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& descriptors) {
        if (descriptors.empty() || keypoints.empty()) {
            return;
        }

        auto& scene = scene_data_[scene_name];
        scene.scene_name = scene_name;
        scene.keypoints_per_image[image_name] = keypoints;
        scene.descriptors_per_image[image_name] = descriptors;
    }

    void registerHomography(const std::string& scene_name,
                           const std::string& from_image,
                           const std::string& to_image,
                           const cv::Mat& homography) {
        if (homography.empty()) {
            return;
        }

        auto& scene = scene_data_[scene_name];
        std::string key = from_image + "_" + to_image;
        scene.homographies[key] = homography;
    }

    void ingestSceneData(SceneData&& scene) {
        if (scene.scene_name.empty()) {
            return;
        }
        scene_data_[scene.scene_name] = std::move(scene);
    }

    void compute(const thesis_project::config::ExperimentConfig& config,
                 const thesis_project::KeypointVerificationParams& params,
                 ::ExperimentMetrics& metrics) {
        if (computed_) {
            return;
        }
        computed_ = true;

        if (scene_data_.empty()) {
            LOG_WARNING("No scene data collected for verification");
            return;
        }

        LOG_INFO("Computing keypoint verification with " +
                 std::to_string(scene_data_.size()) + " scenes");

        std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>> verification_keypoints;
        std::map<std::string, std::map<std::string, cv::Mat>> verification_homographies;

        for (const auto& [scene_name, scene] : scene_data_) {
            for (const auto& [image_name, keypoints] : scene.keypoints_per_image) {
                auto desc_it = scene.descriptors_per_image.find(image_name);
                if (desc_it != scene.descriptors_per_image.end()) {
                    verification_keypoints[scene_name][image_name] =
                        std::make_pair(keypoints, desc_it->second);
                }
            }

            for (const auto& [hom_key, hom_mat] : scene.homographies) {
                auto underscore_pos = hom_key.find('_');
                if (underscore_pos != std::string::npos) {
                    std::string target_image = hom_key.substr(underscore_pos + 1);
                    verification_homographies[scene_name][target_image] = hom_mat;
                }
            }
        }

        auto result = thesis_project::metrics::computeDatasetVerification(
            verification_keypoints,
            verification_homographies,
            params.num_distractor_scenes,
            params.num_distractors_per_scene,
            params.seed);

        metrics.keypoint_verification_ap = result.average_precision;
        metrics.verification_viewpoint_ap = result.viewpoint_ap;
        metrics.verification_illumination_ap = result.illumination_ap;

        LOG_INFO("Verification AP: " + std::to_string(result.average_precision));
        LOG_INFO("  HP-V verification AP: " + std::to_string(result.viewpoint_ap));
        LOG_INFO("  HP-I verification AP: " + std::to_string(result.illumination_ap));
        LOG_INFO("  Total queries: " + std::to_string(result.total_queries));
        LOG_INFO("  Total distractors: " + std::to_string(result.total_distractors));

        scene_data_.clear();
    }

private:
    std::map<std::string, SceneData> scene_data_;
    bool computed_ = false;
};

struct RetrievalAccumulator {
    struct SceneData {
        std::string scene_name;
        std::map<std::string, std::vector<cv::KeyPoint>> keypoints_per_image;
        std::map<std::string, cv::Mat> descriptors_per_image;
        std::map<std::string, cv::Mat> homographies;
    };

    void registerImage(const std::string& scene_name,
                       const std::string& image_name,
                       const std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& descriptors) {
        if (descriptors.empty() || keypoints.empty()) {
            return;
        }

        auto& scene = scene_data_[scene_name];
        scene.scene_name = scene_name;
        scene.keypoints_per_image[image_name] = keypoints;
        scene.descriptors_per_image[image_name] = descriptors;
    }

    void registerHomography(const std::string& scene_name,
                           const std::string& from_image,
                           const std::string& to_image,
                           const cv::Mat& homography) {
        if (homography.empty()) {
            return;
        }

        auto& scene = scene_data_[scene_name];
        std::string key = from_image + "_" + to_image;
        scene.homographies[key] = homography;
    }

    void ingestSceneData(SceneData&& scene) {
        if (scene.scene_name.empty()) {
            return;
        }
        scene_data_[scene.scene_name] = std::move(scene);
    }

    void compute(const thesis_project::config::ExperimentConfig& config,
                 const thesis_project::KeypointRetrievalParams& params,
                 ::ExperimentMetrics& metrics) {
        if (computed_) {
            return;
        }
        computed_ = true;

        if (scene_data_.empty()) {
            LOG_WARNING("No scene data collected for retrieval");
            return;
        }

        LOG_INFO("Computing keypoint retrieval with " +
                 std::to_string(scene_data_.size()) + " scenes");

        std::map<std::string, std::map<std::string, std::pair<std::vector<cv::KeyPoint>, cv::Mat>>> retrieval_keypoints;
        std::map<std::string, std::map<std::string, cv::Mat>> retrieval_homographies;

        for (const auto& [scene_name, scene] : scene_data_) {
            for (const auto& [image_name, keypoints] : scene.keypoints_per_image) {
                auto desc_it = scene.descriptors_per_image.find(image_name);
                if (desc_it != scene.descriptors_per_image.end()) {
                    retrieval_keypoints[scene_name][image_name] =
                        std::make_pair(keypoints, desc_it->second);
                }
            }

            for (const auto& [hom_key, hom_mat] : scene.homographies) {
                auto underscore_pos = hom_key.find('_');
                if (underscore_pos != std::string::npos) {
                    std::string target_image = hom_key.substr(underscore_pos + 1);
                    retrieval_homographies[scene_name][target_image] = hom_mat;
                }
            }
        }

        auto result = thesis_project::metrics::computeDatasetRetrieval(
            retrieval_keypoints,
            retrieval_homographies,
            params.num_distractor_scenes,
            params.num_distractors_per_scene,
            params.seed);

        metrics.keypoint_retrieval_ap = result.average_precision;
        metrics.retrieval_viewpoint_ap = result.viewpoint_ap;
        metrics.retrieval_illumination_ap = result.illumination_ap;
        metrics.retrieval_num_true_positives = result.num_true_positives;
        metrics.retrieval_num_hard_negatives = result.num_hard_negatives;
        metrics.retrieval_num_distractors = result.num_distractors;

        LOG_INFO("Retrieval AP: " + std::to_string(result.average_precision));
        LOG_INFO("  HP-V retrieval AP: " + std::to_string(result.viewpoint_ap));
        LOG_INFO("  HP-I retrieval AP: " + std::to_string(result.illumination_ap));
        LOG_INFO("  Label distribution: TP=" + std::to_string(result.num_true_positives) +
                 " HN=" + std::to_string(result.num_hard_negatives) +
                 " D=" + std::to_string(result.num_distractors));

        scene_data_.clear();
    }

private:
    std::map<std::string, SceneData> scene_data_;
    bool computed_ = false;
};

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

// Create a simple SIFT detector for independent detection
static cv::Ptr<cv::Feature2D> makeDetector(const thesis_project::config::ExperimentConfig& cfg) {
    int maxf = cfg.keypoints.params.max_features;
    if (maxf > 0) return cv::SIFT::create(maxf);
    return cv::SIFT::create();
}

/**
 * @brief Process a single scene (6 images) - THREAD-SAFE for OpenMP parallelization
 */
static std::pair<::ExperimentMetrics, ThreadLocalProfiling> processSingleScene(
    const std::string& scene_folder,
    const std::string& scene_name,
    const SceneProcessingContext& ctx,
    VerificationAccumulator::SceneData* verification_buffer,
    RetrievalAccumulator::SceneData* keypoint_retrieval_buffer
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
                              const std::string& scene_name_inner,
                              const std::string& image_name,
                              std::vector<cv::KeyPoint>& keypoints) {
        const auto det = ensureDetector();
        auto t0 = std::chrono::high_resolution_clock::now();
        det->detect(image, keypoints);
        auto t1 = std::chrono::high_resolution_clock::now();
        prof.detect_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (image_name == "1.ppm") {
            LOG_INFO("Detected " + std::to_string(keypoints.size()) + " keypoints for " + scene_name_inner + "/" + image_name);
        }
    };

    auto loadKeypointsFromDatabase = [&](const std::string& scene_name_inner,
                                         const std::string& image_name,
                                         std::vector<cv::KeyPoint>& keypoints) -> bool {
        if (!ctx.db_ptr) return false;
        auto& db = *ctx.db_ptr;
        bool loaded = false;
        if (ctx.keypoint_set_id >= 0) {
            keypoints = db.getLockedKeypointsFromSet(ctx.keypoint_set_id, scene_name_inner, image_name);
            loaded = !keypoints.empty();
        } else {
            keypoints = db.getLockedKeypoints(scene_name_inner, image_name);
            loaded = !keypoints.empty();
            LOG_INFO("Experiment not using specified keypoint set");
        }

        if (!loaded) {
            LOG_ERROR("No locked keypoints for " + scene_name_inner + "/" + image_name);
            return false;
        }
        return true;
    };

    auto computeDescriptors = [&](const cv::Mat& image,
                                  const std::vector<cv::KeyPoint>& keypoints,
                                  cv::Mat& descriptors,
                                  const std::string& scene_name_inner,
                                  const std::string& image_name) -> bool {
        if (const auto* composite_extractor = dynamic_cast<thesis_project::CompositeDescriptorExtractor*>(&ctx.extractor);
            composite_extractor && composite_extractor->usesPairedKeypointSets()) {
            thesis_project::CompositeDescriptorExtractor::setDatabaseContext(ctx.db_ptr, scene_name_inner, image_name);
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        descriptors = experiment_helpers::computeDescriptorsWithPooling(
            image, keypoints, ctx.extractor, ctx.pooling, ctx.desc_config);
        if (descriptors.empty()) {
            LOG_ERROR("Failed to compute descriptors for " + scene_name_inner + "/" + image_name);
            return false;
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        prof.compute_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return true;
    };

    auto maybeStoreDescriptors = [&](const std::string& scene_name_inner,
                                     const std::string& image_name,
                                     const std::vector<cv::KeyPoint>& keypoints,
                                     const cv::Mat& descriptors) {
        if (!ctx.store_descriptors || descriptors.empty() || !ctx.db_ptr) return;
        if (!ctx.db_ptr->storeDescriptors(ctx.experiment_id, scene_name_inner, image_name,
                                      keypoints, descriptors, ctx.processing_method)) {
            LOG_WARNING("Failed to store descriptors for " + scene_name_inner + "/" + image_name);
        }
    };

    auto maybeStoreMatches = [&](const std::string& scene_name_inner,
                                 const std::string& image_name,
                                 const std::vector<cv::KeyPoint>& keypoints1,
                                 const std::vector<cv::KeyPoint>& keypoints2,
                                 const experiment_helpers::MatchArtifacts& artifacts) {
        if (!ctx.store_matches || artifacts.matches.empty() || !ctx.db_ptr) return;
        if (!ctx.db_ptr->storeMatches(ctx.experiment_id, scene_name_inner, "1.ppm", image_name,
                                  keypoints1, keypoints2, artifacts.matches, artifacts.correctnessFlags)) {
            LOG_WARNING("Failed to store matches for " + scene_name_inner + "/" + image_name);
        }
    };

    auto maybeStoreVisualization = [&](const std::string& scene_name_inner,
                                       const std::string& image_name,
                                       const int image_index,
                                       const cv::Mat& image1,
                                       const cv::Mat& image2,
                                       const std::vector<cv::KeyPoint>& keypoints1,
                                       const std::vector<cv::KeyPoint>& keypoints2,
                                       const experiment_helpers::MatchArtifacts& artifacts) {
        if (!ctx.store_visualizations || artifacts.matches.empty() || !ctx.db_ptr) return;
        cv::Mat match_viz = cli_helpers::generateMatchVisualization(
            image1, image2, keypoints1, keypoints2, artifacts.matches, artifacts.correctnessFlags);
        if (match_viz.empty()) return;

        std::string image_pair = "1_" + std::to_string(image_index);
        double precision = artifacts.matches.empty() ? 0.0 : static_cast<double>(artifacts.correctMatches) / artifacts.matches.size();
        std::string metadata = "{\"matches\":" + std::to_string(artifacts.matches.size()) +
                              ",\"correct\":" + std::to_string(artifacts.correctMatches) +
                              ",\"precision\":" + std::to_string(precision) + "}";

        if (!ctx.db_ptr->storeVisualization(ctx.experiment_id, scene_name_inner, "matches", image_pair, match_viz, metadata)) {
            LOG_WARNING("Failed to store visualization for " + scene_name_inner + "/" + image_pair);
        }
    };

    auto registerSceneImage = [&](auto* buffer,
                                  const std::string& image_name,
                                  const std::vector<cv::KeyPoint>& keypoints,
                                  const cv::Mat& descriptors) {
        if (!buffer || descriptors.empty()) return;
        buffer->scene_name = scene_name;
        buffer->keypoints_per_image[image_name] = keypoints;
        buffer->descriptors_per_image[image_name] = descriptors;
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
    if (!computeDescriptors(image1, keypoints1, descriptors1, scene_name, base_image_name)) {
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
    registerSceneImage(verification_buffer, base_image_name, keypoints1, descriptors1);
    registerSceneImage(keypoint_retrieval_buffer, base_image_name, keypoints1, descriptors1);

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
        if (!computeDescriptors(image2, keypoints2, descriptors2, scene_name, image_name)) {
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

        if (verification_buffer || keypoint_retrieval_buffer) {
            registerSceneImage(verification_buffer, image_name, keypoints2, descriptors2);
            registerSceneImage(keypoint_retrieval_buffer, image_name, keypoints2, descriptors2);
        }

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
        cli_helpers::maybeAccumulateTrueAveragePrecisionFromFile(
            homography_path,
            keypoints1,
            descriptors1,
            keypoints2,
            descriptors2,
            scene_name,
            metrics);

        if (verification_buffer || keypoint_retrieval_buffer) {
            std::ifstream hfile(homography_path);
            if (hfile.good()) {
                cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        hfile >> H.at<double>(r, c);
                    }
                }
                cli_helpers::registerSceneHomography(verification_buffer, base_image_name, image_name, H, scene_name);
                cli_helpers::registerSceneHomography(keypoint_retrieval_buffer, base_image_name, image_name, H, scene_name);
            }
        }
    }

    metrics.calculateMeanPrecision();
    return {metrics, prof};
}

::ExperimentMetrics processDirectoryNew(
    const config::ExperimentConfig& yaml_config,
    const config::ExperimentConfig::DescriptorConfig& desc_config,
    thesis_project::database::DatabaseManager* db_ptr,
    int experiment_id,
    int descriptor_keypoint_set_id,
    ProfilingSummary& profile
) {
    namespace fs = std::filesystem;
    ::ExperimentMetrics overall;
    overall.success = true;

    try {
        if (!fs::exists(yaml_config.dataset.path) || !fs::is_directory(yaml_config.dataset.path)) {
            return ::ExperimentMetrics::createError("Invalid data folder: " + yaml_config.dataset.path);
        }

        const bool use_db_keypoints = db_ptr &&
            (yaml_config.keypoints.params.source == thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION ||
             yaml_config.keypoints.params.source == thesis_project::KeypointSource::INDEPENDENT_DETECTION ||
             yaml_config.keypoints.params.source == thesis_project::KeypointSource::DATABASE);

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
        } else if (desc_config.type == thesis_project::DescriptorType::COMPOSITE) {
            if (desc_config.components.empty()) {
                throw std::runtime_error("composite descriptor requires components in YAML configuration");
            }
            if (desc_config.aggregation_method.empty()) {
                throw std::runtime_error("composite descriptor requires aggregation method in YAML configuration");
            }

            LOG_INFO("Creating CompositeDescriptorExtractor with " +
                     std::to_string(desc_config.components.size()) + " components, aggregation: " +
                     desc_config.aggregation_method);

            std::vector<thesis_project::CompositeDescriptorExtractor::ComponentConfig> component_configs;
            component_configs.reserve(desc_config.components.size());

            for (const auto& comp_desc : desc_config.components) {
                thesis_project::CompositeDescriptorExtractor::ComponentConfig comp_config;
                comp_config.type = comp_desc.type;
                comp_config.weight = comp_desc.weight;
                comp_config.params = comp_desc.params;
                comp_config.keypoint_set_name = comp_desc.keypoint_set_name;
                component_configs.push_back(comp_config);

                std::string kp_info = comp_desc.keypoint_set_name.empty() ?
                    "" : " (keypoint_set: " + comp_desc.keypoint_set_name + ")";
                LOG_INFO("  Component: " + toString(comp_desc.type) +
                         ", weight: " + std::to_string(comp_desc.weight) + kp_info);
            }

            auto aggregation = thesis_project::CompositeDescriptorExtractor::stringToAggregationMethod(
                desc_config.aggregation_method);

            auto output_mode = thesis_project::CompositeDescriptorExtractor::OutputDimensionMode::COLLAPSE_GRAY;
            if (!desc_config.output_dimension.empty()) {
                if (desc_config.output_dimension == "384") {
                    output_mode = thesis_project::CompositeDescriptorExtractor::OutputDimensionMode::PRESERVE_RGB;
                    LOG_INFO("  Output dimension: 384D (PRESERVE_RGB)");
                } else if (desc_config.output_dimension == "128") {
                    output_mode = thesis_project::CompositeDescriptorExtractor::OutputDimensionMode::COLLAPSE_GRAY;
                    LOG_INFO("  Output dimension: 128D (COLLAPSE_GRAY)");
                } else {
                    LOG_WARNING("  Unknown output_dimension '" + desc_config.output_dimension + "', defaulting to 128D");
                }
            }

            extractor = std::make_unique<thesis_project::CompositeDescriptorExtractor>(
                component_configs,
                aggregation,
                output_mode);

            LOG_INFO("CompositeDescriptorExtractor created successfully: " + extractor->name());
        } else {
            extractor = thesis_project::factories::DescriptorFactory::create(desc_config.type);
        }
        const int descriptor_dim = extractor ? extractor->descriptorSize() : 0;
        auto pooling = thesis_project::pooling::PoolingFactory::createFromConfig(desc_config);
        auto matcher = thesis_project::matching::MatchingFactory::createStrategy(
            yaml_config.evaluation.params.matching_method);

        LOG_INFO("Evaluation flags:");
        LOG_INFO("  image_retrieval.enabled = " + std::string(yaml_config.evaluation.params.image_retrieval.enabled ? "true" : "false"));
        LOG_INFO("  keypoint_verification.enabled = " + std::string(yaml_config.evaluation.params.keypoint_verification.enabled ? "true" : "false"));
        LOG_INFO("  keypoint_retrieval.enabled = " + std::string(yaml_config.evaluation.params.keypoint_retrieval.enabled ? "true" : "false"));

        ImageRetrievalAccumulator retrieval_accumulator;
        ImageRetrievalAccumulator* retrieval_ptr = nullptr;
        if (yaml_config.evaluation.params.image_retrieval.enabled) {
            retrieval_ptr = &retrieval_accumulator;
            LOG_INFO("  -> Image retrieval accumulator created");
        }

        VerificationAccumulator verification_accumulator;
        VerificationAccumulator* verification_ptr = nullptr;
        if (yaml_config.evaluation.params.keypoint_verification.enabled) {
            verification_ptr = &verification_accumulator;
            LOG_INFO("  -> Verification accumulator created");
        }

        RetrievalAccumulator keypoint_retrieval_accumulator;
        RetrievalAccumulator* keypoint_retrieval_ptr = nullptr;
        if (yaml_config.evaluation.params.keypoint_retrieval.enabled) {
            keypoint_retrieval_ptr = &keypoint_retrieval_accumulator;
            LOG_INFO("  -> Keypoint retrieval accumulator created");
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
            descriptor_keypoint_set_id,
            store_descriptors,
            store_matches,
            store_visualizations,
            processing_method
        };

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

        const bool retrieval_enabled = yaml_config.evaluation.params.image_retrieval.enabled;
        const bool use_parallel = yaml_config.performance.parallel_scenes &&
                                  !retrieval_enabled;
        int num_threads = yaml_config.performance.num_threads;

#ifdef _OPENMP
        if (retrieval_enabled && yaml_config.performance.parallel_scenes) {
            LOG_INFO("Image retrieval evaluation enabled: processing scenes sequentially for consistency");
        }

        int resolved_threads = num_threads;
        if (resolved_threads <= 0) {
            if (const auto hw_threads = std::thread::hardware_concurrency();
                hw_threads > 0 && hw_threads <= static_cast<unsigned int>(std::numeric_limits<int>::max())){
                resolved_threads = static_cast<int>(hw_threads);
            } else {
                resolved_threads = 4;
            }
        }

        // Apply thread cap globally (scenes and retrieval query-level parallelism use this)
        omp_set_num_threads(resolved_threads);

        if (use_parallel) {
            LOG_INFO("OpenMP enabled: processing " + std::to_string(scenes_to_process.size()) +
                     " scenes with " + std::to_string(resolved_threads) + " threads");
        } else {
            LOG_INFO("Sequential processing: parallel_scenes disabled in configuration "
                     "(threads capped at " + std::to_string(resolved_threads) + " for any internal parallel sections)");
        }
#else
        if (use_parallel) {
            LOG_WARNING("OpenMP not available - running sequentially despite configuration");
        }
#endif

        std::vector<::ExperimentMetrics> scene_metrics(scenes_to_process.size());
        std::vector<ThreadLocalProfiling> scene_profiling(scenes_to_process.size());

        std::vector<VerificationAccumulator::SceneData> verification_buffers;
        if (verification_ptr) {
            verification_buffers.resize(scenes_to_process.size());
        }

        std::vector<RetrievalAccumulator::SceneData> keypoint_retrieval_buffers;
        if (keypoint_retrieval_ptr) {
            keypoint_retrieval_buffers.resize(scenes_to_process.size());
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) if(use_parallel)
#endif
        for (size_t i = 0; i < scenes_to_process.size(); ++i) {
            const auto& [scene_folder, scene_name] = scenes_to_process[i];
            auto result = processSingleScene(
                scene_folder,
                scene_name,
                ctx,
                verification_ptr ? &verification_buffers[i] : nullptr,
                keypoint_retrieval_ptr ? &keypoint_retrieval_buffers[i] : nullptr);
            scene_metrics[i] = result.first;
            scene_profiling[i] = result.second;
        }

        if (verification_ptr) {
            for (auto& buffer : verification_buffers) {
                verification_ptr->ingestSceneData(std::move(buffer));
            }
        }

        if (keypoint_retrieval_ptr) {
            for (auto& buffer : keypoint_retrieval_buffers) {
                keypoint_retrieval_ptr->ingestSceneData(std::move(buffer));
            }
        }

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

        overall.calculateMeanPrecision();

        if (retrieval_ptr) {
            retrieval_ptr->compute(
                yaml_config,
                yaml_config.evaluation.params.image_retrieval.scorer,
                overall);
        }

        if (verification_ptr) {
            verification_ptr->compute(
                yaml_config,
                yaml_config.evaluation.params.keypoint_verification,
                overall);
        }

        if (keypoint_retrieval_ptr) {
            keypoint_retrieval_ptr->compute(
                yaml_config,
                yaml_config.evaluation.params.keypoint_retrieval,
                overall);
        }

        overall.success = true;

        if (yaml_config.evaluation.params.image_retrieval.enabled) {
            LOG_INFO("Image retrieval MAP: " + std::to_string(overall.image_retrieval_map));
        }

        if (yaml_config.evaluation.params.keypoint_verification.enabled) {
            LOG_INFO("Keypoint verification AP: " + std::to_string(overall.keypoint_verification_ap));
            LOG_INFO("  HP-V verification AP: " + std::to_string(overall.verification_viewpoint_ap));
            LOG_INFO("  HP-I verification AP: " + std::to_string(overall.verification_illumination_ap));
        }

        if (yaml_config.evaluation.params.keypoint_retrieval.enabled) {
            LOG_INFO("Keypoint retrieval AP: " + std::to_string(overall.keypoint_retrieval_ap));
            LOG_INFO("  HP-V retrieval AP: " + std::to_string(overall.retrieval_viewpoint_ap));
            LOG_INFO("  HP-I retrieval AP: " + std::to_string(overall.retrieval_illumination_ap));
        }

        auto maybeCleanupEphemeral = [&](const std::string& label,
                                         bool enabled,
                                         bool stored,
                                         const std::function<bool()>& action) {
            if (!enabled || !stored) return;
            if (!db_ptr || experiment_id == -1) return;
            const bool ok = action();
            if (ok) {
                LOG_INFO("Ephemeral cleanup: removed " + label + " for experiment " + std::to_string(experiment_id));
            } else {
                LOG_WARNING("Ephemeral cleanup failed for " + label + " (experiment " +
                            std::to_string(experiment_id) + ")");
            }
        };

        maybeCleanupEphemeral(
            "descriptors",
            yaml_config.database.ephemeral_descriptors,
            store_descriptors,
            [&]() { return db_ptr->deleteDescriptorsForExperiment(experiment_id); });

        maybeCleanupEphemeral(
            "matches",
            yaml_config.database.ephemeral_matches,
            store_matches,
            [&]() { return db_ptr->deleteMatchesForExperiment(experiment_id); });

        profile.detect_ms = detect_ms;
        profile.compute_ms = compute_ms;
        profile.match_ms = match_ms;
        profile.total_images = total_images;
        profile.total_kps = total_kps;
        profile.descriptor_dimension = descriptor_dim;
        return overall;
    } catch (const std::exception& e) {
        return ::ExperimentMetrics::createError(e.what());
    }
}

} // namespace thesis_project::cli::experiment_runner_processing
