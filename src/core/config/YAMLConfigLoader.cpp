#include "YAMLConfigLoader.hpp"
#include "thesis_project/logging.hpp"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <cctype>
#include <vector>

namespace {

std::string trimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool isExplicitAssignmentToken(const std::string& raw_value) {
    const auto normalized = toLowerCopy(trimCopy(raw_value));
    return normalized == "independent" || normalized == "none";
}

bool descriptorRequiresColor(thesis_project::DescriptorType type) {
    using thesis_project::DescriptorType;
    switch (type) {
        case DescriptorType::RGBSIFT:
        case DescriptorType::RGBSIFT_CHANNEL_AVG:
        case DescriptorType::HoNC:
        case DescriptorType::DSPRGBSIFT_V2:
        case DescriptorType::DSPHONC_V2:
            return true;
        default:
            return false;
    }
}

std::string formatAvailableSets(const std::unordered_set<std::string>& sets) {
    if (sets.empty()) {
        return "<none>";
    }
    std::vector<std::string> sorted(sets.begin(), sets.end());
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream oss;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << sorted[i];
    }
    return oss.str();
}

}


namespace thesis_project::config {

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

        if (root["performance"]) {
            parsePerformance(root["performance"], config.performance);
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

        const bool explicit_token = isExplicitAssignmentToken(keypoints.params.keypoint_set_name);
        if (explicit_token) {
            // Clear sentinel values so we don't treat them as actual DB set names later
            keypoints.params.keypoint_set_name.clear();
        }

        const bool database_requires_explicit =
            keypoints.params.source == KeypointSource::DATABASE &&
            keypoints.params.keypoint_set_name.empty();

        keypoints.assignment_mode = (explicit_token || database_requires_explicit)
            ? KeypointAssignmentMode::EXPLICIT_ONLY
            : KeypointAssignmentMode::INHERIT_FROM_PRIMARY;

        // Parse alternative keypoint sets
        if (node["alternative_keypoints"] && node["alternative_keypoints"].IsSequence()) {
            keypoints.alternative_keypoints.clear();
            for (const auto& alt_node : node["alternative_keypoints"]) {
                ExperimentConfig::Keypoints::AlternativeKeypointSet alt_set;

                if (alt_node["keypoint_set_name"]) {
                    alt_set.keypoint_set_name = alt_node["keypoint_set_name"].as<std::string>();
                } else {
                    throw std::runtime_error("Alternative keypoint set must have 'keypoint_set_name' field");
                }

                if (alt_node["description"]) {
                    alt_set.description = alt_node["description"].as<std::string>();
                }

                keypoints.alternative_keypoints.push_back(alt_set);
            }
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

            // Parse keypoint set override (optional)
            if (desc_node["keypoint_set_name"]) {
                desc_config.keypoint_set_name = desc_node["keypoint_set_name"].as<std::string>();
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

            if (desc_node["extended"]) {
                desc_config.params.surf_extended = desc_node["extended"].as<bool>();
            }

            bool use_color_specified = false;
            bool use_color_value = false;
            if (desc_node["use_color"]) {
                use_color_specified = true;
                use_color_value = desc_node["use_color"].as<bool>();
                desc_config.params.use_color = use_color_value;
            }

            if (desc_config.type != DescriptorType::NONE && descriptorRequiresColor(desc_config.type)) {
                if (use_color_specified && !use_color_value) {
                    throw std::runtime_error(
                        "Descriptor '" + desc_config.name + "' of type '" +
                        thesis_project::toString(desc_config.type) +
                        "' requires use_color: true, but the configuration set use_color: false."
                    );
                }
                if (!desc_config.params.use_color) {
                    desc_config.params.use_color = true;
                    LOG_WARNING("Descriptor '" + desc_config.name + "' (" +
                                thesis_project::toString(desc_config.type) +
                                ") requires color input; automatically enabling use_color: true.");
                }
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

            // VGG descriptor config (optional)
            if (desc_node["vgg"]) {
                const auto& vgg = desc_node["vgg"];
                if (vgg["desc_type"]) {
                    int dt = vgg["desc_type"].as<int>();
                    // Validate VGG descriptor type (100=VGG_120, 101=VGG_80, 102=VGG_64, 103=VGG_48)
                    if (dt >= 100 && dt <= 103) {
                        desc_config.params.vgg_desc_type = dt;
                    }
                }
                if (vgg["isigma"]) desc_config.params.vgg_isigma = vgg["isigma"].as<float>();
                if (vgg["img_normalize"]) desc_config.params.vgg_img_normalize = vgg["img_normalize"].as<bool>();
                if (vgg["use_scale_orientation"]) desc_config.params.vgg_use_scale_orientation = vgg["use_scale_orientation"].as<bool>();
                if (vgg["scale_factor"]) desc_config.params.vgg_scale_factor = vgg["scale_factor"].as<float>();
                if (vgg["dsc_normalize"]) desc_config.params.vgg_dsc_normalize = vgg["dsc_normalize"].as<bool>();
            }

            // Composite descriptor configuration (optional)
            if (desc_node["aggregation"]) {
                desc_config.aggregation_method = desc_node["aggregation"].as<std::string>();
            }

            if (desc_node["output_dimension"]) {
                desc_config.output_dimension = desc_node["output_dimension"].as<std::string>();
            } else {
                // Default to 128D for channel_wise fusion
                desc_config.output_dimension = "128";
            }

            if (desc_node["weight"]) {
                desc_config.weight = desc_node["weight"].as<double>();
            }

            if (desc_node["components"] && desc_node["components"].IsSequence()) {
                // Recursively parse component descriptors
                desc_config.components.clear();
                bool composite_needs_color = false;

                for (const auto& comp_node : desc_node["components"]) {
                    ExperimentConfig::DescriptorConfig comp_config;
                    comp_config.type = DescriptorType::NONE;

                    // Parse component type (required)
                    if (comp_node["descriptor"]) {
                        comp_config.type = stringToDescriptorType(comp_node["descriptor"].as<std::string>());
                    } else if (comp_node["type"]) {
                        comp_config.type = stringToDescriptorType(comp_node["type"].as<std::string>());
                    } else {
                        throw std::runtime_error("Component descriptor must have 'descriptor' or 'type' field");
                    }

                    bool comp_use_color_specified = false;
                    bool comp_use_color_value = false;

                    // Parse component weight (optional)
                    if (comp_node["weight"]) {
                        comp_config.weight = comp_node["weight"].as<double>();
                    }

                    // Parse component keypoint set override (optional)
                    if (comp_node["keypoint_set_name"]) {
                        comp_config.keypoint_set_name = comp_node["keypoint_set_name"].as<std::string>();
                    }

                    // Parse component-specific params (optional)
                    if (comp_node["device"]) {
                        comp_config.params.device = comp_node["device"].as<std::string>();
                    }
                    if (comp_node["use_color"]) {
                        comp_use_color_specified = true;
                        comp_use_color_value = comp_node["use_color"].as<bool>();
                        comp_config.params.use_color = comp_use_color_value;
                    }

                    // SURF extended flag (128D descriptors)
                    if (comp_node["extended"]) {
                        comp_config.params.surf_extended = comp_node["extended"].as<bool>();
                    }
                    // Add more param parsing here if needed

                    if (comp_config.type != DescriptorType::NONE && descriptorRequiresColor(comp_config.type)) {
                        if (comp_use_color_specified && !comp_use_color_value) {
                            throw std::runtime_error(
                                "Composite descriptor '" + desc_config.name + "' component '" +
                                thesis_project::toString(comp_config.type) +
                                "' requires use_color: true, but the configuration set use_color: false."
                            );
                        }
                        if (!comp_config.params.use_color) {
                            comp_config.params.use_color = true;
                            LOG_WARNING("Composite descriptor '" + desc_config.name + "' component '" +
                                        thesis_project::toString(comp_config.type) +
                                        "' requires color input; automatically enabling use_color: true.");
                        }
                    }

                    if (comp_config.params.use_color) {
                        composite_needs_color = true;
                    }

                    desc_config.components.push_back(comp_config);
                }

                // Validate composite descriptor has components
                if (desc_config.type == DescriptorType::COMPOSITE && desc_config.components.empty()) {
                    throw std::runtime_error(
                        "Composite descriptor '" + desc_config.name + "' must have at least 2 components"
                    );
                }

                // IMPORTANT: Set use_color=true for composite if ANY component needs color
                // This prevents grayscale conversion that would create fake color (R=G=B)
                if (desc_config.type == DescriptorType::COMPOSITE && composite_needs_color) {
                    if (use_color_specified && !use_color_value) {
                        throw std::runtime_error(
                            "Composite descriptor '" + desc_config.name +
                            "' has color-dependent components; set use_color: true."
                        );
                    }
                    if (!desc_config.params.use_color) {
                        desc_config.params.use_color = true;
                        LOG_WARNING("Composite descriptor '" + desc_config.name +
                                    "' requires color input because one or more components need color; "
                                    "automatically enabling use_color: true.");
                    }
                }
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

        // Validate keypoint set assignments
        const bool explicit_assignment_mode =
            config.keypoints.assignment_mode == KeypointAssignmentMode::EXPLICIT_ONLY;

        // Build set of all valid keypoint set names (primary + alternatives)
        std::unordered_set<std::string> valid_keypoint_sets;

        // Add primary keypoint set if specified and we're in inheritance mode
        if (!explicit_assignment_mode && !config.keypoints.params.keypoint_set_name.empty()) {
            valid_keypoint_sets.insert(config.keypoints.params.keypoint_set_name);
        }

        // Add alternative keypoint sets
        for (const auto& alt : config.keypoints.alternative_keypoints) {
            if (alt.keypoint_set_name.empty()) {
                throw std::runtime_error("YAML validation error: alternative_keypoints entry must have keypoint_set_name");
            }
            if (!valid_keypoint_sets.insert(alt.keypoint_set_name).second) {
                throw std::runtime_error(
                    "YAML validation error: duplicate keypoint set name in alternatives: " + alt.keypoint_set_name
                );
            }
        }
        auto ensureKeypointSetKnown = [&](const std::string& context, const std::string& set_name) {
            if (valid_keypoint_sets.empty()) {
                throw std::runtime_error(
                    "YAML validation error: " + context +
                    " references keypoint_set_name '" + set_name +
                    "' but no keypoint sets are defined (set keypoints.keypoint_set_name or add alternative_keypoints)."
                );
            }
            if (!valid_keypoint_sets.count(set_name)) {
                throw std::runtime_error(
                    "YAML validation error: " + context +
                    " references unknown keypoint_set_name '" + set_name +
                    "'. Available sets: " + formatAvailableSets(valid_keypoint_sets)
                );
            }
        };

        // Validate descriptor keypoint set overrides
        for (const auto& desc : config.descriptors) {
            const bool descriptor_is_composite = desc.type == DescriptorType::COMPOSITE;
            const bool descriptor_has_override = !desc.keypoint_set_name.empty();

            if (descriptor_has_override) {
                ensureKeypointSetKnown("Descriptor '" + desc.name + "'", desc.keypoint_set_name);
            } else if (explicit_assignment_mode && !descriptor_is_composite) {
                throw std::runtime_error(
                    "YAML validation error: Descriptor '" + desc.name +
                    "' must specify keypoint_set_name because keypoints.keypoint_set_name "
                    "is set to an explicit assignment flag."
                );
            }

            if (descriptor_is_composite) {
                bool all_components_have_sets = true;
                for (size_t i = 0; i < desc.components.size(); ++i) {
                    if (const auto& comp = desc.components[i]; !comp.keypoint_set_name.empty()) {
                        ensureKeypointSetKnown(
                            "Composite descriptor '" + desc.name + "' component " + std::to_string(i),
                            comp.keypoint_set_name
                        );
                    } else {
                        all_components_have_sets = false;
                    }
                }

                if (explicit_assignment_mode && !descriptor_has_override && !all_components_have_sets) {
                    throw std::runtime_error(
                        "YAML validation error: Composite descriptor '" + desc.name +
                        "' must specify keypoint_set_name on the descriptor or on every component "
                        "because keypoints.keypoint_set_name enforces explicit assignments."
                    );
                }
            }
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
                if (const auto norm_str = matching["norm"].as<std::string>(); norm_str == "l1") evaluation.params.norm_type = cv::NORM_L1;
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

        // Parse image retrieval evaluation parameters
        if (node["image_retrieval"]) {
            const auto& retrieval = node["image_retrieval"];

            if (retrieval["enabled"]) {
                evaluation.params.image_retrieval.enabled = retrieval["enabled"].as<bool>();
            }

            if (retrieval["scorer"]) {
                evaluation.params.image_retrieval.scorer = retrieval["scorer"].as<std::string>();
            }
        }

        // Parse keypoint verification evaluation parameters (Bojanic et al. 2020)
        if (node["keypoint_verification"]) {
            const auto& verification = node["keypoint_verification"];

            if (verification["enabled"]) {
                evaluation.params.keypoint_verification.enabled = verification["enabled"].as<bool>();
            }

            if (verification["num_distractor_scenes"]) {
                evaluation.params.keypoint_verification.num_distractor_scenes =
                    verification["num_distractor_scenes"].as<int>();
            }

            if (verification["num_distractors_per_scene"]) {
                evaluation.params.keypoint_verification.num_distractors_per_scene =
                    verification["num_distractors_per_scene"].as<int>();
            }

            if (verification["seed"]) {
                evaluation.params.keypoint_verification.seed = verification["seed"].as<int>();
            }
        }

        if (node["keypoint_retrieval"]) {
            const auto& retrieval = node["keypoint_retrieval"];

            if (retrieval["enabled"]) {
                evaluation.params.keypoint_retrieval.enabled = retrieval["enabled"].as<bool>();
            }

            if (retrieval["num_distractor_scenes"]) {
                evaluation.params.keypoint_retrieval.num_distractor_scenes =
                    retrieval["num_distractor_scenes"].as<int>();
            }

            if (retrieval["num_distractors_per_scene"]) {
                evaluation.params.keypoint_retrieval.num_distractors_per_scene =
                    retrieval["num_distractors_per_scene"].as<int>();
            }

            if (retrieval["seed"]) {
                evaluation.params.keypoint_retrieval.seed = retrieval["seed"].as<int>();
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

    void YAMLConfigLoader::parsePerformance(const YAML::Node& node, PerformanceParams& performance) {
        if (node["num_threads"]) performance.num_threads = node["num_threads"].as<int>();
        if (node["parallel_scenes"]) performance.parallel_scenes = node["parallel_scenes"].as<bool>();
        if (node["parallel_images"]) performance.parallel_images = node["parallel_images"].as<bool>();
        if (node["batch_size"]) performance.batch_size = node["batch_size"].as<int>();
        if (node["enable_profiling"]) performance.enable_profiling = node["enable_profiling"].as<bool>();
    }

    // Migration removed
    
    // Type conversion helper methods
    DescriptorType YAMLConfigLoader::stringToDescriptorType(const std::string& str) {
        if (str == "sift") return DescriptorType::SIFT;
        if (str == "rgbsift") return DescriptorType::RGBSIFT;
        if (str == "rgbsift_channel_avg") return DescriptorType::RGBSIFT_CHANNEL_AVG;
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
        if (str == "composite") return DescriptorType::COMPOSITE;
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
        if (str == "surf") return KeypointGenerator::SURF;
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
            if (desc.params.surf_extended) {
                out << YAML::Key << "extended" << YAML::Value << true;
            }
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

} // namespace thesis_project::config
