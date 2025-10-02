#pragma once

#include "ExperimentConfig.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>

namespace thesis_project {
namespace config {

    /**
     * @brief YAML configuration loader
     *
     * Loads experiment configurations from YAML files into
     * strongly-typed configuration structures.
     */
    class YAMLConfigLoader {
    public:
        /**
         * @brief Load experiment configuration from YAML file
         * @param yaml_path Path to YAML file
         * @return Loaded configuration
         * @throws std::runtime_error if file cannot be loaded or parsed
         */
        static ExperimentConfig loadFromFile(const std::string& yaml_path);

        /**
         * @brief Load experiment configuration from YAML string
         * @param yaml_content YAML content as string
         * @return Loaded configuration
         */
        static ExperimentConfig loadFromString(const std::string& yaml_content);

        /**
         * @brief Save experiment configuration to YAML file
         * @param config Configuration to save
         * @param yaml_path Output file path
         */
        static void saveToFile(const ExperimentConfig& config, const std::string& yaml_path);

    private:
        // Internal helper for loading from YAML node
        static ExperimentConfig loadFromYAML(const YAML::Node& root);

        // Helper methods for parsing different sections
        static void parseExperiment(const YAML::Node& node, ExperimentConfig::Experiment& experiment);
        static void parseDataset(const YAML::Node& node, ExperimentConfig::Dataset& dataset);
        static void parseKeypoints(const YAML::Node& node, ExperimentConfig::Keypoints& keypoints);
        static void parseDescriptors(const YAML::Node& node, std::vector<ExperimentConfig::DescriptorConfig>& descriptors);
        static void parseEvaluation(const YAML::Node& node, ExperimentConfig::Evaluation& evaluation);
        static void parseOutput(const YAML::Node& node, ExperimentConfig::Output& output);
        static void parseDatabase(const YAML::Node& node, DatabaseParams& database);
        // Migration removed in Schema v1

        // Type conversion helpers
        static DescriptorType stringToDescriptorType(const std::string& str);
        static PoolingStrategy stringToPoolingStrategy(const std::string& str);
        static PoolingAggregation stringToPoolingAggregation(const std::string& str);
        static RootingStage stringToRootingStage(const std::string& str);
        static KeypointGenerator stringToKeypointGenerator(const std::string& str);
        static MatchingMethod stringToMatchingMethod(const std::string& str);
        static ValidationMethod stringToValidationMethod(const std::string& str);

        // Basic schema/range validation
        static void validate(const ExperimentConfig& config);
    };

} // namespace config
} // namespace thesis_project
