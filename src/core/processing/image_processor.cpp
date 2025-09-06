#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <future>
#include <string>
#include <chrono>

#include "../config/experiment_config.hpp"
#include "image_processor.hpp"
#include "../keypoints/locked_in_keypoints.hpp"
#ifdef BUILD_DATABASE
#include "thesis_project/database/DatabaseManager.hpp"
#endif
#include "thesis_project/types.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/metrics/MetricsCalculator.hpp"
#include "src/core/metrics/TrueAveragePrecision.hpp"

namespace fs = boost::filesystem;
// TODO: Add another method to visual check the locked in keypoints

// Helper functions for descriptive database strings (research-friendly)
std::string descriptorTypeToDescriptiveString(DescriptorType type) {
    switch (type) {
        case DESCRIPTOR_SIFT: return "sift";
        case DESCRIPTOR_vSIFT: return "vanilla_sift";
        case DESCRIPTOR_RGBSIFT: return "rgb_sift";
        case DESCRIPTOR_HoNC: return "histogram_of_normalized_colors";
        case NO_DESCRIPTOR: return "no_descriptor";
        default: return "unknown_descriptor";
    }
}

std::string poolingStrategyToDescriptiveString(PoolingStrategy strategy) {
    switch (strategy) {
        case NONE: return "no_pooling";
        case DOMAIN_SIZE_POOLING: return "domain_size_pooling";
        case STACKING: return "stacking";
        default: return "unknown_pooling";
    }
}

std::string normalizationStageToDescriptiveString(NormalizationStage stage) {
    switch (stage) {
        case BEFORE_POOLING: return "before_pooling";
        case AFTER_POOLING: return "after_pooling";
        case NO_NORMALIZATION: return "no_normalization";
        default: return "unknown_normalization";
    }
}

std::string rootingStageToDescriptiveString(RootingStage stage) {
    switch (stage) {
        case R_BEFORE_POOLING: return "rooting_before_pooling";
        case R_AFTER_POOLING: return "rooting_after_pooling";
        case R_NONE: return "no_rooting";
        default: return "unknown_rooting";
    }
}

std::string descriptorColorSpaceToDescriptiveString(DescriptorColorSpace colorSpace) {
    switch (colorSpace) {
        case D_COLOR: return "color";
        case D_BW: return "grayscale";
        default: return "unknown_color_space";
    }
}

std::string normTypeToDescriptiveString(int normType) {
    switch (normType) {
        case cv::NORM_L1: return "l1_norm";
        case cv::NORM_L2: return "l2_norm";
        default: return "unknown_norm_type";
    }
}


ExperimentMetrics image_processor::process_directory(const std::string &data_folder, const std::string &results_folder, const experiment_config &config) {
    ExperimentMetrics overall_metrics;
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        if (!fs::exists(data_folder) || !fs::is_directory(data_folder)) {
            std::cerr << "Invalid data folder: " << data_folder << std::endl;
            return ExperimentMetrics::createError("Invalid data folder: " + data_folder);
        }
        
        std::vector<std::future<ExperimentMetrics>> futures;
        std::vector<ExperimentMetrics> folder_metrics;

        for (const auto &entry : fs::directory_iterator(data_folder)) {
            const auto &path = entry.path();
            if (fs::is_directory(path)) {
                std::string subfolder = path.filename().string();
                std::string results_subfolder = results_folder;
                results_subfolder.append("/").append(subfolder);

                // Asynchronous execution path using std::async
                if (config.useMultiThreading) {
                    auto fut = std::async(std::launch::async, [&, path, results_subfolder, subfolder]() -> ExperimentMetrics {
                        try {
                            if(config.descriptorOptions.UseLockedInKeypoints) {
                                return process_image_folder_keypoints_locked(path.string(), results_subfolder, config);
                            } else {
                                return process_image_folder_keypoints_unlocked(path.string(), results_subfolder, config);
                            }
                        } catch (const std::exception &e) {
                            std::cerr << "Exception in async task: " << e.what() << std::endl;
                            return ExperimentMetrics::createError("Exception in async task: " + std::string(e.what()));
                        } catch (...) {
                            std::cerr << "Unknown exception in async task" << std::endl;
                            return ExperimentMetrics::createError("Unknown exception in async task");
                        }
                    });
                    futures.push_back(std::move(fut));
                }
                    // Synchronous (sequential) execution path
                else {
                    try {
                        ExperimentMetrics folder_result;
                        if (config.verificationType == MATCHES) {
                            VisualVerification::verifyMatches(path.string(), results_subfolder, config);
                            // Visual verification doesn't return metrics, mark as successful but no data
                            folder_result = ExperimentMetrics::createSuccess();
                        } else if (config.verificationType == HOMOGRAPHY) {
                            VisualVerification::verifyHomography(path.string(), results_subfolder, config);
                            // Visual verification doesn't return metrics, mark as successful but no data
                            folder_result = ExperimentMetrics::createSuccess();
                        } else {
                            if(config.descriptorOptions.UseLockedInKeypoints) {
                                folder_result = process_image_folder_keypoints_locked(path.string(), results_subfolder, config);
                            } else {
                                folder_result = process_image_folder_keypoints_unlocked(path.string(), results_subfolder, config);
                            }
                        }
                        folder_metrics.push_back(folder_result);
                    } catch (const std::exception &e) {
                        std::cerr << "Exception in synchronous task: " << e.what() << std::endl;
                        folder_metrics.push_back(ExperimentMetrics::createError("Exception in synchronous task: " + std::string(e.what())));
                    } catch (...) {
                        std::cerr << "Unknown exception in synchronous task" << std::endl;
                        folder_metrics.push_back(ExperimentMetrics::createError("Unknown exception in synchronous task"));
                    }
                }
            }
        }

        // Wait for all asynchronous tasks to complete and collect results
        if (config.useMultiThreading) {
            for (auto &fut : futures) {
                try {
                    ExperimentMetrics result = fut.get(); // This blocks until the future's task is complete
                    folder_metrics.push_back(result);
                } catch (const std::exception &e) {
                    std::cerr << "Exception caught during future.get(): " << e.what() << std::endl;
                    folder_metrics.push_back(ExperimentMetrics::createError("Exception during future.get(): " + std::string(e.what())));
                } catch (...) {
                    std::cerr << "Unknown exception caught during future.get()" << std::endl;
                    folder_metrics.push_back(ExperimentMetrics::createError("Unknown exception during future.get()"));
                }
            }
        }

        // Aggregate all folder metrics using MetricsCalculator
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time_ms = MetricsCalculator::calculateProcessingTime(start_time, end_time);
        overall_metrics = MetricsCalculator::aggregateMetrics(folder_metrics, processing_time_ms);

        return overall_metrics;
    } catch (const std::exception &e) {
        std::cerr << "Error processing directory: " << e.what() << std::endl;
        return ExperimentMetrics::createError("Error processing directory: " + std::string(e.what()));
    }
}

ExperimentMetrics image_processor::process_image_folder_keypoints_locked(const std::string &folder, const std::string &results_folder,
                                                            const experiment_config &config) {
    ExperimentMetrics metrics;
    std::string scene_name = boost::filesystem::path(folder).filename().string();
    std::cout << "Processing locked folder: " << folder << std::endl;
    // Results folder creation removed - using database storage only

    // Read in 1.ppm as the base image
    cv::Mat image1 = cv::imread(folder + "/1.ppm", cv::IMREAD_COLOR);

    if (image1.empty()) {
        std::cerr << "Failed to read image: " << folder + "/1.ppm" << std::endl;
        metrics.success = false;
        metrics.error_message = "Failed to read image: " + folder + "/1.ppm";
        return metrics;
    }

    // Apply color space conversion if needed (consistent with unlocked path)
    if (config.descriptorOptions.descriptorColorSpace == D_BW) {
        cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    }

    // Load locked-in keypoints for image 1
    std::vector<cv::KeyPoint> keypoints1;
    
#ifdef BUILD_DATABASE
    // Load keypoints from database only (no CSV fallback)
    thesis_project::database::DatabaseManager db("experiments.db", true);
    keypoints1 = db.getLockedKeypoints(scene_name, "1.ppm");
    
    if (keypoints1.empty()) {
        std::cerr << "[ERROR] No keypoints found in database for " << scene_name << "/1.ppm. Use keypoint_manager to generate keypoints first." << std::endl;
        return ExperimentMetrics(); // Return empty metrics
    }
    std::cout << "[INFO] Loaded " << keypoints1.size() << " keypoints from database for " << scene_name << "/1.ppm" << std::endl;
#else
    std::cerr << "[ERROR] Database support not enabled. Build with -DBUILD_DATABASE=ON" << std::endl;
    return ExperimentMetrics(); // Return empty metrics
#endif

    // Compute descriptors for image 1 using the locked-in keypoints
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> result1 = processor_utils::detectAndComputeWithConfigLocked(image1, keypoints1, config);
    cv::Mat descriptors1 = result1.second;

#ifdef BUILD_DATABASE
    // Store descriptors in database for research analysis
    if (db.isEnabled() && !descriptors1.empty() && config.experiment_id != -1) {
        // Build descriptive processing method string for database queries
        std::string processing_method = descriptorTypeToDescriptiveString(config.descriptorOptions.descriptorType) + "_" +
                                       descriptorColorSpaceToDescriptiveString(config.descriptorOptions.descriptorColorSpace) + "_" +
                                       poolingStrategyToDescriptiveString(config.descriptorOptions.poolingStrategy) + "_" +
                                       normalizationStageToDescriptiveString(config.descriptorOptions.normalizationStage) + "_" +
                                       rootingStageToDescriptiveString(config.descriptorOptions.rootingStage);
        
        // Store descriptors with experiment linking and descriptive metadata
        db.storeDescriptors(config.experiment_id, scene_name, "1.ppm", keypoints1, descriptors1, processing_method,
                           normalizationStageToDescriptiveString(config.descriptorOptions.normalizationStage),
                           rootingStageToDescriptiveString(config.descriptorOptions.rootingStage),
                           poolingStrategyToDescriptiveString(config.descriptorOptions.poolingStrategy));
    }
#endif

    // CSV outputs disabled - using database storage only

    for (int i = 2; i <= 6; i++) {
        std::string imagePath = folder + "/" + std::to_string(i) + ".ppm";
        cv::Mat image2 = cv::imread(imagePath, cv::IMREAD_COLOR);

        if (image2.empty()) continue;

        // Apply same color space conversion as image1
        if (config.descriptorOptions.descriptorColorSpace == D_BW) {
            cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
        }

        // Load locked-in keypoints for image i
        std::vector<cv::KeyPoint> keypoints2;
        std::string image_name = std::to_string(i) + ".ppm";
        
#ifdef BUILD_DATABASE
        // Load keypoints from database only (no CSV fallback)
        keypoints2 = db.getLockedKeypoints(scene_name, image_name);
        
        if (keypoints2.empty()) {
            std::cerr << "[ERROR] No keypoints found in database for " << scene_name << "/" << image_name << ". Use keypoint_manager to generate keypoints first." << std::endl;
            continue; // Skip this image
        }
        std::cout << "[INFO] Loaded " << keypoints2.size() << " keypoints from database for " << scene_name << "/" << image_name << std::endl;
#else
        std::cerr << "[ERROR] Database support not enabled. Build with -DBUILD_DATABASE=ON" << std::endl;
        return ExperimentMetrics(); // Return empty metrics
#endif

        // Compute descriptors for image i using the locked-in keypoints
        std::pair<std::vector<cv::KeyPoint>, cv::Mat> result2 = processor_utils::detectAndComputeWithConfigLocked(image2, keypoints2, config);
        cv::Mat descriptors2 = result2.second;

#ifdef BUILD_DATABASE
        // Store descriptors in database for research analysis
        if (db.isEnabled() && !descriptors2.empty() && config.experiment_id != -1) {
            // Build descriptive processing method string (same as image 1)
            std::string processing_method = descriptorTypeToDescriptiveString(config.descriptorOptions.descriptorType) + "_" +
                                           descriptorColorSpaceToDescriptiveString(config.descriptorOptions.descriptorColorSpace) + "_" +
                                           poolingStrategyToDescriptiveString(config.descriptorOptions.poolingStrategy) + "_" +
                                           normalizationStageToDescriptiveString(config.descriptorOptions.normalizationStage) + "_" +
                                           rootingStageToDescriptiveString(config.descriptorOptions.rootingStage);
            
            // Store descriptors with experiment linking and descriptive metadata
            db.storeDescriptors(config.experiment_id, scene_name, image_name, keypoints2, descriptors2, processing_method,
                               normalizationStageToDescriptiveString(config.descriptorOptions.normalizationStage),
                               rootingStageToDescriptiveString(config.descriptorOptions.rootingStage),
                               poolingStrategyToDescriptiveString(config.descriptorOptions.poolingStrategy));
        }
#endif

        // CSV outputs disabled - using database storage only

        // Match descriptors
        std::vector<cv::DMatch> matches = processor_utils::matchDescriptors(descriptors1, descriptors2, config.matchingStrategy);

        // Read homography matrix for true mAP computation
        std::string homographyFilename = folder + "/H_1_" + std::to_string(i);
        cv::Mat homography_1_to_i;
        try {
            homography_1_to_i = processor_utils::readHomography(homographyFilename);
        } catch (const std::exception& e) {
            // Continue with legacy evaluation only
        }

        /*
         * Dual evaluation system: Legacy precision + True IR-style mAP
         * 
         * Legacy Method: Index correspondence check (backward compatibility)
         * Since keypoints2 are transformed versions of keypoints1, we expect match.queryIdx == match.trainIdx
         * 
         * True mAP Method: Homography-based geometric correctness (research standard)
         * For each query keypoint in image1, compute Average Precision using homography-based ground truth
         */

        // Legacy evaluation using known keypoint correspondences
        int correctMatches = 0;
        for (const auto& match : matches) {
            if (match.queryIdx == match.trainIdx) {
                correctMatches++;
            }
        }
        double precision = matches.empty() ? 0.0 : static_cast<double>(correctMatches) / matches.size();

        // True IR-style mAP computation (if homography available)
        if (!homography_1_to_i.empty() && !keypoints1.empty() && !keypoints2.empty()) {
            
            // TRUE IR-style mAP: Evaluate ALL queries against ALL targets
            for (int query_idx = 0; query_idx < static_cast<int>(keypoints1.size()); ++query_idx) {
                // Compute descriptor distances from this query to ALL target keypoints
                std::vector<double> all_distances;
                cv::Mat query_descriptor = descriptors1.row(query_idx);
                
                // Skip queries with invalid descriptors (rare but possible)
                if (query_descriptor.empty() || cv::norm(query_descriptor) == 0.0) {
                    // Add dummy result for excluded query
                    TrueAveragePrecision::QueryAPResult dummy_result;
                    dummy_result.ap = 0.0;
                    dummy_result.has_potential_match = false;
                    metrics.addQueryAP(scene_name, dummy_result);
                    continue;
                }
                
                for (int target_idx = 0; target_idx < static_cast<int>(keypoints2.size()); ++target_idx) {
                    cv::Mat target_descriptor = descriptors2.row(target_idx);
                    
                    // Check for valid descriptors (guard against NaN/empty)
                    if (query_descriptor.empty() || target_descriptor.empty()) {
                        all_distances.push_back(std::numeric_limits<double>::infinity());
                        continue;
                    }
                    
                    // Compute L2SQR distance for speed (ranking unchanged)
                    double distance = cv::norm(query_descriptor, target_descriptor, cv::NORM_L2SQR);
                    all_distances.push_back(distance);
                }
                
                // Compute Average Precision for this query against ALL targets
                auto ap_result = TrueAveragePrecision::computeQueryAP(
                    keypoints1[query_idx],     // Query keypoint in image 1
                    homography_1_to_i,        // Homography from image 1 to image i
                    keypoints2,               // ALL keypoints in image i  
                    all_distances,            // Descriptor distances to ALL targets
                    3.0                       // Default pixel tolerance (τ=3.0px)
                );
                
                // Add to metrics accumulation
                metrics.addQueryAP(scene_name, ap_result);
            }
        }

        // Capture metrics for database storage
        metrics.addImageResult(scene_name, precision, matches.size(), keypoints2.size());
    }

    // Finalize metrics calculation
    metrics.calculateMeanPrecision();
    metrics.success = true;
    return metrics;
}

ExperimentMetrics image_processor::process_image_folder_keypoints_unlocked(const std::string &folder, const std::string &results_folder,
                                                              const experiment_config &config) {
    ExperimentMetrics metrics;
    std::string scene_name = boost::filesystem::path(folder).filename().string();
    std::cout << "Processing folder: " << folder << std::endl;
    // Results folder creation removed - using database storage only

    // Read in 1.ppm as the base image
    cv::Mat image1 = cv::imread(folder + "/1.ppm", cv::IMREAD_COLOR);

    if (config.descriptorOptions.descriptorColorSpace == D_BW) {
        // convert image1 to greyscale
        cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
    }

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> result1 = processor_utils::detectAndComputeWithConfig(image1, config);
    std::vector<cv::KeyPoint> keypoints1 = result1.first;
    cv::Mat descriptors1 = result1.second;

    // CSV outputs disabled - using database storage only

    for (int i = 2; i <= 6; i++) {
        std::string imagePath = folder + "/" + std::to_string(i) + ".ppm";
        cv::Mat image2 = cv::imread(imagePath, cv::IMREAD_COLOR);

        if (image2.empty()) continue;

        // Apply same color conversion as image1 BEFORE descriptor computation
        if (config.descriptorOptions.descriptorColorSpace == D_BW) {
            cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
        }

        std::pair<std::vector<cv::KeyPoint>, cv::Mat> result2 = processor_utils::detectAndComputeWithConfig(image2, config);
        std::vector<cv::KeyPoint> keypoints2 = result2.first;
        cv::Mat descriptors2 = result2.second;

        // CSV outputs disabled - using database storage only

        // Match normal descriptors
        std::vector<cv::DMatch> matches = processor_utils::matchDescriptors(descriptors1, descriptors2, config.matchingStrategy);
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
        std::vector<cv::DMatch> topMatches = std::vector<cv::DMatch>(matches.begin(), matches.begin() + std::min(2000, (int)matches.size()));

        // Legacy precision calculation
        int correctMatches = 0;
        for (const auto& match : matches) {
            if (match.queryIdx == match.trainIdx) {
                correctMatches++;
            }
        }
        double precision = matches.empty() ? 0.0 : static_cast<double>(correctMatches) / matches.size();

        // Read homography matrix for true mAP computation
        std::string homographyFilename = folder + "/H_1_" + std::to_string(i);
        cv::Mat homography_1_to_i;
        try {
            homography_1_to_i = processor_utils::readHomography(homographyFilename);
        } catch (const std::exception& e) {
            // Continue with legacy evaluation only
        }

        // True IR-style mAP computation (if homography available)
        if (!homography_1_to_i.empty() && !keypoints1.empty() && !keypoints2.empty()) {
            
            // TRUE IR-style mAP: Evaluate ALL queries against ALL targets
            for (int query_idx = 0; query_idx < static_cast<int>(keypoints1.size()); ++query_idx) {
                // Compute descriptor distances from this query to ALL target keypoints
                std::vector<double> all_distances;
                cv::Mat query_descriptor = descriptors1.row(query_idx);
                
                // Skip queries with invalid descriptors (rare but possible)
                if (query_descriptor.empty() || cv::norm(query_descriptor) == 0.0) {
                    // Add dummy result for excluded query
                    TrueAveragePrecision::QueryAPResult dummy_result;
                    dummy_result.ap = 0.0;
                    dummy_result.has_potential_match = false;
                    metrics.addQueryAP(scene_name, dummy_result);
                    continue;
                }
                
                for (int target_idx = 0; target_idx < static_cast<int>(keypoints2.size()); ++target_idx) {
                    cv::Mat target_descriptor = descriptors2.row(target_idx);
                    
                    // Check for valid descriptors (guard against NaN/empty)
                    if (query_descriptor.empty() || target_descriptor.empty()) {
                        all_distances.push_back(std::numeric_limits<double>::infinity());
                        continue;
                    }
                    
                    // Compute L2SQR distance for speed (ranking unchanged)
                    double distance = cv::norm(query_descriptor, target_descriptor, cv::NORM_L2SQR);
                    all_distances.push_back(distance);
                }
                
                // Compute Average Precision for this query against ALL targets
                auto ap_result = TrueAveragePrecision::computeQueryAP(
                    keypoints1[query_idx],     // Query keypoint in image 1
                    homography_1_to_i,        // Homography from image 1 to image i
                    keypoints2,               // ALL keypoints in image i  
                    all_distances,            // Descriptor distances to ALL targets
                    3.0                       // Default pixel tolerance (τ=3.0px)
                );
                
                // Add to metrics accumulation
                metrics.addQueryAP(scene_name, ap_result);
            }
        }

        // Capture metrics
        metrics.addImageResult(scene_name, precision, matches.size(), keypoints2.size());

        // Results stored in database only

    }

    // Finalize metrics calculation
    metrics.calculateMeanPrecision();
    metrics.success = true;
    return metrics;
}

