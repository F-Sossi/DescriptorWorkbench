#include "YAMLConfigLoader.hpp"
#include "thesis_project/logging.hpp"
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace thesis_project {
namespace config {

    ExperimentConfig YAMLConfigLoader::loadFromFile(const std::string& yaml_path) {
        try {
            YAML::Node root = YAML::LoadFile(yaml_path);
            return loadFromYAML(root);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("YAML parsing error in " + yaml_path + ": " + e.what());
        } catch (const std::exception& e) {
            throw std::runtime_error("Error loading " + yaml_path + ": " + e.what());
        }
    }
    
    ExperimentConfig YAMLConfigLoader::loadFromString(const std::string& yaml_content) {
        try {
            YAML::Node root = YAML::Load(yaml_content);
            return loadFromYAML(root);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
        }
    }
    
    ExperimentConfig YAMLConfigLoader::loadFromYAML(const YAML::Node& root) {
        ExperimentConfig config;
        
        // Parse each section
        if (root["experiment"]) {
            parseExperiment(root["experiment"], config.experiment);
        }
        
        if (root["dataset"]) {
            parseDataset(root["dataset"], config.dataset);
        }
        
        if (root["keypoints"]) {
            parseKeypoints(root["keypoints"], config.keypoints);
        }
        
        if (root["descriptors"]) {
            parseDescriptors(root["descriptors"], config.descriptors);
        }
        
        if (root["evaluation"]) {
            parseEvaluation(root["evaluation"], config.evaluation);
        }
        
        if (root["database"]) {
            parseDatabase(root["database"], config.database);
        }
        // Migration removed: ignore any 'migration' key silently
        
        // Basic validation
        validate(config);
        
        return config;
    }
    
    void YAMLConfigLoader::parseExperiment(const YAML::Node& node, ExperimentConfig::Experiment& experiment) {
        if (node["name"]) experiment.name = node["name"].as<std::string>();
        if (node["description"]) experiment.description = node["description"].as<std::string>();
        if (node["version"]) experiment.version = node["version"].as<std::string>();
        if (node["author"]) experiment.author = node["author"].as<std::string>();
    }
    
    void YAMLConfigLoader::parseDataset(const YAML::Node& node, ExperimentConfig::Dataset& dataset) {
        if (node["type"]) dataset.type = node["type"].as<std::string>();
        if (node["path"]) dataset.path = node["path"].as<std::string>();
        
        if (node["scenes"] && node["scenes"].IsSequence()) {
            dataset.scenes.clear();
            for (const auto& scene : node["scenes"]) {
                dataset.scenes.push_back(scene.as<std::string>());
            }
        }
    }
    
    void YAMLConfigLoader::parseKeypoints(const YAML::Node& node, ExperimentConfig::Keypoints& keypoints) {
        if (node["generator"]) {
            keypoints.generator = stringToKeypointGenerator(node["generator"].as<std::string>());
        }
        
        // Parse keypoint parameters
        if (node["max_features"]) {
            keypoints.params.max_features = node["max_features"].as<int>();
        }
        if (node["contrast_threshold"]) {
            keypoints.params.contrast_threshold = node["contrast_threshold"].as<float>();
        }
        if (node["edge_threshold"]) {
            keypoints.params.edge_threshold = node["edge_threshold"].as<float>();
        }
        if (node["sigma"]) {
            keypoints.params.sigma = node["sigma"].as<float>();
        }
        if (node["num_octaves"]) {
            keypoints.params.num_octaves = node["num_octaves"].as<int>();
        }
        if (node["use_locked_keypoints"]) {
            keypoints.params.use_locked_keypoints = node["use_locked_keypoints"].as<bool>();
        }
        if (node["source"]) {
            keypoints.params.source = keypointSourceFromString(node["source"].as<std::string>());
        }
        if (node["keypoint_set_name"]) {
            keypoints.params.keypoint_set_name = node["keypoint_set_name"].as<std::string>();
        }
        if (node["locked_keypoints_path"]) {
            keypoints.params.locked_keypoints_path = node["locked_keypoints_path"].as<std::string>();
        }
    }
    
    void YAMLConfigLoader::parseDescriptors(const YAML::Node& node, std::vector<ExperimentConfig::DescriptorConfig>& descriptors) {
        if (!node.IsSequence()) {
            throw std::runtime_error("Descriptors section must be a sequence");
        }
        
        descriptors.clear();
        for (const auto& desc_node : node) {
            ExperimentConfig::DescriptorConfig desc_config;
            // Initialize type to NONE for validation clarity
            desc_config.type = DescriptorType::NONE;
            
            if (desc_node["name"]) {
                desc_config.name = desc_node["name"].as<std::string>();
            }
            
            if (desc_node["type"]) {
                desc_config.type = stringToDescriptorType(desc_node["type"].as<std::string>());
            }
            
            // Parse descriptor parameters
            if (desc_node["pooling"]) {
                desc_config.params.pooling = stringToPoolingStrategy(desc_node["pooling"].as<std::string>());
            }

            if (desc_node["pooling_aggregation"]) {
                desc_config.params.pooling_aggregation =
                    stringToPoolingAggregation(desc_node["pooling_aggregation"].as<std::string>());
            }

            if (desc_node["scales"] && desc_node["scales"].IsSequence()) {
                desc_config.params.scales.clear();
                for (const auto& scale : desc_node["scales"]) {
                    desc_config.params.scales.push_back(scale.as<float>());
                }
            }
            if (desc_node["scale_weights"] && desc_node["scale_weights"].IsSequence()) {
                desc_config.params.scale_weights.clear();
                for (const auto& w : desc_node["scale_weights"]) {
                    desc_config.params.scale_weights.push_back(w.as<float>());
                }
            }
            if (desc_node["scale_weighting"]) {
                std::string wt = desc_node["scale_weighting"].as<std::string>();
                if (wt == "gaussian") desc_config.params.scale_weighting = ScaleWeighting::GAUSSIAN;
                else if (wt == "triangular") desc_config.params.scale_weighting = ScaleWeighting::TRIANGULAR;
                else desc_config.params.scale_weighting = ScaleWeighting::UNIFORM;
            }
            if (desc_node["scale_weight_sigma"]) {
                desc_config.params.scale_weight_sigma = desc_node["scale_weight_sigma"].as<float>();
            }
            
            if (desc_node["normalize_before_pooling"]) {
                desc_config.params.normalize_before_pooling = desc_node["normalize_before_pooling"].as<bool>();
            }
            
            if (desc_node["normalize_after_pooling"]) {
                desc_config.params.normalize_after_pooling = desc_node["normalize_after_pooling"].as<bool>();
            }

            if (desc_node["rooting_stage"]) {
                desc_config.params.rooting_stage =
                    stringToRootingStage(desc_node["rooting_stage"].as<std::string>());
            }

            if (desc_node["use_color"]) {
                desc_config.params.use_color = desc_node["use_color"].as<bool>();
            }

            if (desc_node["device"]) {
                desc_config.params.device = desc_node["device"].as<std::string>();
            }
            
            if (desc_node["norm_type"]) {
                std::string norm_str = desc_node["norm_type"].as<std::string>();
                if (norm_str == "l1") desc_config.params.norm_type = cv::NORM_L1;
                else if (norm_str == "l2") desc_config.params.norm_type = cv::NORM_L2;
                else desc_config.params.norm_type = cv::NORM_L2; // default
            }
            
            if (desc_node["secondary_descriptor"]) {
                desc_config.params.secondary_descriptor = stringToDescriptorType(desc_node["secondary_descriptor"].as<std::string>());
            }
            
            if (desc_node["stacking_weight"]) {
                desc_config.params.stacking_weight = desc_node["stacking_weight"].as<float>();
            }

            // DNN patch descriptor config (optional)
            if (desc_node["dnn"]) {
                const auto& dnn = desc_node["dnn"];
                if (dnn["model"]) desc_config.params.dnn_model_path = dnn["model"].as<std::string>();
                if (dnn["input_size"]) desc_config.params.dnn_input_size = dnn["input_size"].as<int>();
                if (dnn["support_multiplier"]) desc_config.params.dnn_support_multiplier = dnn["support_multiplier"].as<float>();
                if (dnn["rotate_to_upright"]) desc_config.params.dnn_rotate_upright = dnn["rotate_to_upright"].as<bool>();
                if (dnn["mean"]) desc_config.params.dnn_mean = dnn["mean"].as<float>();
                if (dnn["std"]) desc_config.params.dnn_std = dnn["std"].as<float>();
                if (dnn["per_patch_standardize"]) desc_config.params.dnn_per_patch_standardize = dnn["per_patch_standardize"].as<bool>();
            }
            
            descriptors.push_back(desc_config);
        }
    }

    void YAMLConfigLoader::validate(const ExperimentConfig& config) {
        // Required dataset path
        if (config.dataset.path.empty()) {
            throw std::runtime_error("YAML validation error: dataset.path is required");
        }

        // Must have at least one descriptor
        if (config.descriptors.empty()) {
            throw std::runtime_error("YAML validation error: descriptors list must not be empty");
        }

        // Ensure descriptor names are unique
        std::unordered_set<std::string> names;
        
        // Validate each descriptor
        for (const auto& d : config.descriptors) {
            if (d.name.empty()) {
                throw std::runtime_error("YAML validation error: descriptor.name is required");
            }
            if (!names.insert(d.name).second) {
                throw std::runtime_error("YAML validation error: descriptor.name must be unique: " + d.name);
            }
            if (d.type == DescriptorType::NONE) {
                throw std::runtime_error("YAML validation error: descriptor.type is required for " + d.name);
            }
            if (d.params.stacking_weight < 0.0f || d.params.stacking_weight > 1.0f) {
                throw std::runtime_error("YAML validation error: stacking_weight must be in [0,1] for " + d.name);
            }

            // Stacking requires a secondary descriptor to be specified (cannot be NONE)
            if (d.params.pooling == PoolingStrategy::STACKING) {
                if (d.params.secondary_descriptor == DescriptorType::NONE) {
                    throw std::runtime_error("YAML validation error: stacking requires secondary_descriptor for " + d.name);
                }
            }

            // DSP and scale semantics
            if (!d.params.scales.empty()) {
                for (float s : d.params.scales) {
                    if (s <= 0.0f) {
                        throw std::runtime_error("YAML validation error: all scales must be > 0 for " + d.name);
                    }
                }
            }
            if (!d.params.scale_weights.empty()) {
                if (d.params.scale_weights.size() != d.params.scales.size()) {
                    throw std::runtime_error("YAML validation error: scale_weights length must match scales for " + d.name);
                }
            }
            if (d.params.scale_weight_sigma <= 0.0f) {
                throw std::runtime_error("YAML validation error: scale_weight_sigma must be > 0 for " + d.name);
            }

            // Warnings
            if (d.params.pooling == PoolingStrategy::NONE && !d.params.scales.empty()) {
                LOG_WARNING("Pooling is 'none' but scales were provided for descriptor '" + d.name + "' — scales will be ignored.");
            }
            if (!d.params.scale_weights.empty() && d.params.scale_weighting != ScaleWeighting::UNIFORM) {
                LOG_WARNING("Both scale_weights and scale_weighting specified for descriptor '" + d.name + "' — explicit weights take precedence.");
            }
        }

        // Keypoint parameter sanity checks
        if (config.keypoints.params.max_features < 0) {
            throw std::runtime_error("YAML validation error: keypoints.max_features must be >= 0");
        }
        if (config.keypoints.params.num_octaves <= 0) {
            throw std::runtime_error("YAML validation error: keypoints.num_octaves must be > 0");
        }
        if (config.keypoints.params.sigma <= 0.0f) {
            throw std::runtime_error("YAML validation error: keypoints.sigma must be > 0");
        }

        // Evaluation threshold typical range [0,1]
        if (config.evaluation.params.match_threshold < 0.0f || config.evaluation.params.match_threshold > 1.0f) {
            throw std::runtime_error("YAML validation error: evaluation.matching.threshold must be in [0,1]");
        }
    }
    
    void YAMLConfigLoader::parseEvaluation(const YAML::Node& node, ExperimentConfig::Evaluation& evaluation) {
        // Parse matching parameters
        if (node["matching"]) {
            const auto& matching = node["matching"];
            
            if (matching["method"]) {
                evaluation.params.matching_method = stringToMatchingMethod(matching["method"].as<std::string>());
            }
            
            if (matching["norm"]) {
                std::string norm_str = matching["norm"].as<std::string>();
                if (norm_str == "l1") evaluation.params.norm_type = cv::NORM_L1;
                else if (norm_str == "l2") evaluation.params.norm_type = cv::NORM_L2;
                else evaluation.params.norm_type = cv::NORM_L2; // default
            }
            
            if (matching["cross_check"]) {
                evaluation.params.cross_check = matching["cross_check"].as<bool>();
            }
            
            if (matching["threshold"]) {
                evaluation.params.match_threshold = matching["threshold"].as<float>();
            }

            // Support ratio_threshold as an alias for threshold when using ratio_test
            if (matching["ratio_threshold"]) {
                evaluation.params.match_threshold = matching["ratio_threshold"].as<float>();
            }
        }
        
        // Parse validation parameters
        if (node["validation"]) {
            const auto& validation = node["validation"];
            
            if (validation["method"]) {
                evaluation.params.validation_method = stringToValidationMethod(validation["method"].as<std::string>());
            }
            
            if (validation["threshold"]) {
                evaluation.params.validation_threshold = validation["threshold"].as<float>();
            }
            
            if (validation["min_matches"]) {
                evaluation.params.min_matches_for_homography = validation["min_matches"].as<int>();
            }
        }
        
        // Legacy keys removed in Schema v1 (no parsing of matching_threshold / validation_method)
    }
    
    void YAMLConfigLoader::parseDatabase(const YAML::Node& node, DatabaseParams& database) {
        if (node["connection"]) database.connection_string = node["connection"].as<std::string>();
        if (node["save_keypoints"]) database.save_keypoints = node["save_keypoints"].as<bool>();
        if (node["save_descriptors"]) database.save_descriptors = node["save_descriptors"].as<bool>();
        if (node["save_matches"]) database.save_matches = node["save_matches"].as<bool>();
        if (node["save_visualizations"]) database.save_visualizations = node["save_visualizations"].as<bool>();
    }

    // Migration removed
    
    // Type conversion helper methods
    DescriptorType YAMLConfigLoader::stringToDescriptorType(const std::string& str) {
        if (str == "sift") return DescriptorType::SIFT;
        if (str == "rgbsift") return DescriptorType::RGBSIFT;
        if (str == "vsift" || str == "vanilla_sift") return DescriptorType::vSIFT;
        if (str == "honc") return DescriptorType::HoNC;
        if (str == "dnn_patch") return DescriptorType::DNN_PATCH;
        if (str == "vgg") return DescriptorType::VGG;
        if (str == "dspsift") return DescriptorType::DSPSIFT;
        if (str == "dspsift_v2") return DescriptorType::DSPSIFT_V2;
        if (str == "dsprgbsift_v2") return DescriptorType::DSPRGBSIFT_V2;
        if (str == "dsphowh_v2") return DescriptorType::DSPHOWH_V2;
        if (str == "dsphonc_v2") return DescriptorType::DSPHONC_V2;
        if (str == "libtorch_hardnet") return DescriptorType::LIBTORCH_HARDNET;
        if (str == "libtorch_sosnet") return DescriptorType::LIBTORCH_SOSNET;
        if (str == "libtorch_l2net") return DescriptorType::LIBTORCH_L2NET;
        if (str == "orb") return DescriptorType::ORB;
        if (str == "surf") return DescriptorType::SURF;
        throw std::runtime_error("Unknown descriptor type: " + str);
    }
    
    PoolingStrategy YAMLConfigLoader::stringToPoolingStrategy(const std::string& str) {
        if (str == "none") return PoolingStrategy::NONE;
        if (str == "domain_size_pooling" || str == "dsp") return PoolingStrategy::DOMAIN_SIZE_POOLING;
        if (str == "stacking") return PoolingStrategy::STACKING;
        throw std::runtime_error("Unknown pooling strategy: " + str);
    }

    PoolingAggregation YAMLConfigLoader::stringToPoolingAggregation(const std::string& str) {
        if (str == "average" || str == "avg") return PoolingAggregation::AVERAGE;
        if (str == "max") return PoolingAggregation::MAX;
        if (str == "min") return PoolingAggregation::MIN;
        if (str == "concatenate" || str == "concat") return PoolingAggregation::CONCATENATE;
        if (str == "weighted_avg" || str == "weighted") return PoolingAggregation::WEIGHTED_AVG;
        // Default to average (don't throw error for backward compatibility)
        return PoolingAggregation::AVERAGE;
    }

    RootingStage YAMLConfigLoader::stringToRootingStage(const std::string& str) {
        if (str == "before_pooling" || str == "before") return RootingStage::R_BEFORE_POOLING;
        if (str == "after_pooling" || str == "after") return RootingStage::R_AFTER_POOLING;
        if (str == "none" || str.empty()) return RootingStage::R_NONE;
        // Default to none (don't throw error for backward compatibility)
        return RootingStage::R_NONE;
    }

    KeypointGenerator YAMLConfigLoader::stringToKeypointGenerator(const std::string& str) {
        if (str == "sift") return KeypointGenerator::SIFT;
        if (str == "harris") return KeypointGenerator::HARRIS;
        if (str == "orb") return KeypointGenerator::ORB;
        if (str == "keynet") return KeypointGenerator::KEYNET;
        if (str == "locked_in") return KeypointGenerator::LOCKED_IN;
        throw std::runtime_error("Unknown keypoint generator: " + str);
    }
    
    MatchingMethod YAMLConfigLoader::stringToMatchingMethod(const std::string& str) {
        if (str == "brute_force") return MatchingMethod::BRUTE_FORCE;
        if (str == "flann") return MatchingMethod::FLANN;
        if (str == "ratio_test") return MatchingMethod::RATIO_TEST;
        throw std::runtime_error("Unknown matching method: " + str);
    }
    
    ValidationMethod YAMLConfigLoader::stringToValidationMethod(const std::string& str) {
        if (str == "homography") return ValidationMethod::HOMOGRAPHY;
        if (str == "cross_image") return ValidationMethod::CROSS_IMAGE;
        if (str == "none") return ValidationMethod::NONE;
        throw std::runtime_error("Unknown validation method: " + str);
    }
    
    void YAMLConfigLoader::saveToFile(const ExperimentConfig& config, const std::string& yaml_path) {
        YAML::Emitter out;
        
        out << YAML::BeginMap;
        
        // Experiment section
        out << YAML::Key << "experiment";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "name" << YAML::Value << config.experiment.name;
        out << YAML::Key << "description" << YAML::Value << config.experiment.description;
        out << YAML::Key << "version" << YAML::Value << config.experiment.version;
        out << YAML::Key << "author" << YAML::Value << config.experiment.author;
        out << YAML::EndMap;
        
        // Dataset section
        out << YAML::Key << "dataset";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "type" << YAML::Value << config.dataset.type;
        out << YAML::Key << "path" << YAML::Value << config.dataset.path;
        out << YAML::Key << "scenes" << YAML::Value << config.dataset.scenes;
        out << YAML::EndMap;
        
        // Keypoints section
        out << YAML::Key << "keypoints";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "generator" << YAML::Value << toString(config.keypoints.generator);
        out << YAML::Key << "max_features" << YAML::Value << config.keypoints.params.max_features;
        out << YAML::Key << "contrast_threshold" << YAML::Value << config.keypoints.params.contrast_threshold;
        out << YAML::Key << "edge_threshold" << YAML::Value << config.keypoints.params.edge_threshold;
        out << YAML::EndMap;
        
        // Descriptors section
        out << YAML::Key << "descriptors";
        out << YAML::Value << YAML::BeginSeq;
        for (const auto& desc : config.descriptors) {
            out << YAML::BeginMap;
            out << YAML::Key << "name" << YAML::Value << desc.name;
            out << YAML::Key << "type" << YAML::Value << toString(desc.type);
            out << YAML::Key << "pooling" << YAML::Value << toString(desc.params.pooling);
            if (!desc.params.scales.empty()) {
                out << YAML::Key << "scales" << YAML::Value << desc.params.scales;
            }
            out << YAML::Key << "normalize_after_pooling" << YAML::Value << desc.params.normalize_after_pooling;
            out << YAML::Key << "use_color" << YAML::Value << desc.params.use_color;
            if (desc.params.device != "auto") {
                out << YAML::Key << "device" << YAML::Value << desc.params.device;
            }
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        
        // Evaluation section
        out << YAML::Key << "evaluation";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "matching" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "method" << YAML::Value << toString(config.evaluation.params.matching_method);
        // Use ratio_threshold for ratio test, threshold for others
        if (config.evaluation.params.matching_method == MatchingMethod::RATIO_TEST) {
            out << YAML::Key << "ratio_threshold" << YAML::Value << config.evaluation.params.match_threshold;
        } else {
            out << YAML::Key << "threshold" << YAML::Value << config.evaluation.params.match_threshold;
        }
        out << YAML::Key << "cross_check" << YAML::Value << config.evaluation.params.cross_check;
        out << YAML::EndMap;
        out << YAML::Key << "validation" << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "method" << YAML::Value << toString(config.evaluation.params.validation_method);
        out << YAML::Key << "threshold" << YAML::Value << config.evaluation.params.validation_threshold;
        out << YAML::EndMap;
        out << YAML::EndMap;
        
        // Database section
        out << YAML::Key << "database";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "connection" << YAML::Value << config.database.connection_string;
        out << YAML::Key << "save_keypoints" << YAML::Value << config.database.save_keypoints;
        out << YAML::Key << "save_descriptors" << YAML::Value << config.database.save_descriptors;
        out << YAML::Key << "save_matches" << YAML::Value << config.database.save_matches;
        out << YAML::Key << "save_visualizations" << YAML::Value << config.database.save_visualizations;
        out << YAML::EndMap;
        
        out << YAML::EndMap;
        
        // Write to file
        std::ofstream file(yaml_path);
        file << out.c_str();
    }

} // namespace config
} // namespace thesis_project
