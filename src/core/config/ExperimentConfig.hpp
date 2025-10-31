#pragma once

#include "thesis_project/types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace thesis_project::config {

    /**
     * @brief Modern experiment configuration using Stage 2 types
     *
     * This replaces the old experiment_config class with a modern,
     * YAML-configurable system using our type-safe enums.
     */
    struct ExperimentConfig {
        // Experiment metadata
        struct Experiment {
            std::string name;
            std::string description;
            std::string version = "1.0";
            std::string author;
        } experiment;

        // Dataset configuration
        struct Dataset {
            std::string type = "hpatches";
            std::string path = "data/hpatches/";
            std::vector<std::string> scenes;  // Empty = all scenes
        } dataset;

        // Keypoint detection configuration
        struct Keypoints {
            KeypointGenerator generator = KeypointGenerator::SIFT;
            KeypointParams params;
        } keypoints;

        // Descriptor configurations (can have multiple for comparison)
        struct DescriptorConfig {
            std::string name;
            DescriptorType type;
            DescriptorParams params;

            // Composite descriptor support
            std::vector<DescriptorConfig> components;  // Component descriptors for COMPOSITE type
            std::string aggregation_method;            // "average", "weighted_avg", "max", "min", "concatenate"
            double weight = 1.0;                       // Weight for this component in weighted averaging
        };
        std::vector<DescriptorConfig> descriptors;

        // Evaluation configuration
        struct Evaluation {
            EvaluationParams params;
        } evaluation;

        // Database configuration
        DatabaseParams database;

        // Performance configuration
        PerformanceParams performance;

        // Migration removed: new pipeline is the default
    };

}
