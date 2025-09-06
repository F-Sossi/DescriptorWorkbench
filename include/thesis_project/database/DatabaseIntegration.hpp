#ifndef THESIS_PROJECT_DATABASE_INTEGRATION_HPP
#define THESIS_PROJECT_DATABASE_INTEGRATION_HPP

#include "thesis_project/database/DatabaseManager.hpp"
#include "../../../src/core/config/experiment_config.hpp"

namespace thesis_project {
namespace database {

/**
 * @brief Helper functions to integrate existing experiment_config with database
 */
namespace integration {

    /**
     * @brief Convert existing experiment_config to database format
     * @param config Existing experiment configuration
     * @return Database-compatible experiment configuration
     */
    inline ExperimentConfig toDbConfig(const experiment_config& config) {
        ExperimentConfig db_config;

        // Convert descriptor type (using actual enum values from your system)
        if (config.descriptorOptions.descriptorType == DESCRIPTOR_SIFT) {
            db_config.descriptor_type = "SIFT";
        } else if (config.descriptorOptions.descriptorType == DESCRIPTOR_RGBSIFT) {
            db_config.descriptor_type = "RGBSIFT";
        } else if (config.descriptorOptions.descriptorType == DESCRIPTOR_HoNC) {
            db_config.descriptor_type = "HoNC";
        } else if (config.descriptorOptions.descriptorType == DESCRIPTOR_vSIFT) {
            db_config.descriptor_type = "vSIFT";
        } else {
            db_config.descriptor_type = "UNKNOWN";
        }

        // Convert pooling strategy (using actual enum values from your system)
        if (config.descriptorOptions.poolingStrategy == STACKING) {
            db_config.pooling_strategy = "STACKING";
        } else if (config.descriptorOptions.poolingStrategy == DOMAIN_SIZE_POOLING) {
            db_config.pooling_strategy = "DOMAIN_SIZE_POOLING";
        } else if (config.descriptorOptions.poolingStrategy == NONE) {
            db_config.pooling_strategy = "NONE";
        } else {
            db_config.pooling_strategy = "NONE";
        }

        // Set other parameters with sensible defaults
        db_config.max_features = 1000; // Default since maxFeatures doesn't exist in DescriptorOptions
        db_config.similarity_threshold = config.matchThreshold;

        // Add additional parameters as string map
        db_config.parameters["normType"] = std::to_string(config.descriptorOptions.normType);
        db_config.parameters["useMultiThreading"] = config.useMultiThreading ? "true" : "false";

        if (config.descriptorOptions.imageType == COLOR) {
            db_config.parameters["imageType"] = "COLOR";
        } else if (config.descriptorOptions.imageType == BW) {
            db_config.parameters["imageType"] = "BW";
        }

        if (config.descriptorOptions.descriptorColorSpace == D_COLOR) {
            db_config.parameters["descriptorColorSpace"] = "D_COLOR";
        } else if (config.descriptorOptions.descriptorColorSpace == D_BW) {
            db_config.parameters["descriptorColorSpace"] = "D_BW";
        }

        return db_config;
    }

    /**
     * @brief Create database results structure
     * @param experiment_id ID from recorded configuration
     * @param descriptor_name Name of descriptor used
     * @param dataset_name Name of dataset processed
     * @param map_score Mean Average Precision score
     * @param processing_time Processing time in milliseconds
     * @return Database-compatible results structure
     */
    inline ExperimentResults createDbResults(
        int experiment_id,
        const std::string& descriptor_name,
        const std::string& dataset_name,
        double map_score,
        double processing_time = 0.0) {

        ExperimentResults results;
        results.experiment_id = experiment_id;
        results.descriptor_type = descriptor_name;
        results.dataset_name = dataset_name;
        results.mean_average_precision = map_score;
        results.processing_time_ms = processing_time;

        // Initialize other metrics with defaults
        results.precision_at_1 = 0.0;
        results.precision_at_5 = 0.0;
        results.recall_at_1 = 0.0;
        results.recall_at_5 = 0.0;
        results.total_matches = 0;
        results.total_keypoints = 0;

        return results;
    }

    /**
     * @brief Enhanced results creation with full metrics
     * @param experiment_id ID from recorded configuration
     * @param descriptor_name Name of descriptor used
     * @param dataset_name Name of dataset processed
     * @param map_score Mean Average Precision score
     * @param precision_1 Precision at rank 1
     * @param precision_5 Precision at rank 5
     * @param recall_1 Recall at rank 1
     * @param recall_5 Recall at rank 5
     * @param total_matches Total number of matches
     * @param total_keypoints Total keypoints detected
     * @param processing_time Processing time in milliseconds
     * @return Database-compatible results structure
     */
    inline ExperimentResults createDetailedDbResults(
        int experiment_id,
        const std::string& descriptor_name,
        const std::string& dataset_name,
        double map_score,
        double precision_1,
        double precision_5,
        double recall_1,
        double recall_5,
        int total_matches,
        int total_keypoints,
        double processing_time = 0.0) {

        ExperimentResults results;
        results.experiment_id = experiment_id;
        results.descriptor_type = descriptor_name;
        results.dataset_name = dataset_name;
        results.mean_average_precision = map_score;
        results.precision_at_1 = precision_1;
        results.precision_at_5 = precision_5;
        results.recall_at_1 = recall_1;
        results.recall_at_5 = recall_5;
        results.total_matches = total_matches;
        results.total_keypoints = total_keypoints;
        results.processing_time_ms = processing_time;

        return results;
    }

} // namespace integration
} // namespace database
} // namespace thesis_project

#endif // THESIS_PROJECT_DATABASE_INTEGRATION_HPP
