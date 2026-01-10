/**
 * @file patch_benchmark.cpp
 * @brief CLI for evaluating descriptors on pre-extracted HPatches patches
 *
 * This tool evaluates descriptor fusion strategies directly on HPatches patches,
 * removing the keypoint location quality confound from intersection experiments.
 *
 * Usage:
 *   ./patch_benchmark --patches ../hpatches-release --descriptor hardnet
 *   ./patch_benchmark --patches ../hpatches-release --fusion "hardnet+sosnet" --method concat
 *   ./patch_benchmark config.yaml
 */

#include "core/patches/PatchLoader.hpp"
#include "core/patches/PatchDescriptorExtractor.hpp"
#include "core/patches/PatchDescriptorFactory.hpp"
#include "core/benchmark/HPatchesBenchmark.hpp"
#include "core/benchmark/PatchMetrics.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <thesis_project/types.hpp>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cctype>

// Simple argument parser
struct Args {
    std::string patches_dir = "../hpatches-release";
    std::string descriptor;
    std::vector<std::string> fusion_components;
    std::string fusion_method = "concatenate";
    std::string device = "auto";
    bool easy = true;
    bool hard = true;
    bool tough = true;
    bool verbose = true;
    bool help = false;
    std::string config_file;
};

struct DescriptorConfig {
    std::string name;
    std::string type;
    std::vector<std::string> components;
    std::string method = "concatenate";
    std::vector<float> weights;
    std::string device = "auto";

    bool isFusion() const {
        return !components.empty() || type == "composite" || type == "fusion";
    }
};

struct BenchmarkConfig {
    thesis_project::benchmark::HPatchesBenchmark::Config benchmark;
    std::vector<DescriptorConfig> descriptors;
    bool save_to_database = false;
};

class NamedPatchDescriptorExtractor final : public thesis_project::patches::IPatchDescriptorExtractor {
public:
    NamedPatchDescriptorExtractor(
        std::unique_ptr<thesis_project::patches::IPatchDescriptorExtractor> inner,
        std::string name)
        : inner_(std::move(inner)),
          name_(std::move(name)) {}

    cv::Mat extractFromPatches(
        const std::vector<cv::Mat>& patches,
        const thesis_project::DescriptorParams& params) override {
        return inner_->extractFromPatches(patches, params);
    }

    std::string name() const override { return name_; }
    int descriptorSize() const override { return inner_->descriptorSize(); }
    int descriptorType() const override { return inner_->descriptorType(); }
    bool requiresResize() const override { return inner_->requiresResize(); }
    int expectedPatchSize() const override { return inner_->expectedPatchSize(); }
    std::unique_ptr<thesis_project::patches::IPatchDescriptorExtractor> clone() const override {
        return std::make_unique<NamedPatchDescriptorExtractor>(inner_->clone(), name_);
    }

private:
    std::unique_ptr<thesis_project::patches::IPatchDescriptorExtractor> inner_;
    std::string name_;
};

void printUsage(const char* prog) {
    std::cout << "HPatches Patch Benchmark\n";
    std::cout << "========================\n\n";
    std::cout << "Evaluate descriptor fusion on pre-extracted HPatches patches.\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " [options]\n";
    std::cout << "  " << prog << " <config.yaml>\n\n";
    std::cout << "Options:\n";
    std::cout << "  --patches <dir>       Path to hpatches-release directory (default: ../hpatches-release)\n";
    std::cout << "  --descriptor <name>   Single descriptor: sift, hardnet, sosnet, rgbsift, etc.\n";
    std::cout << "  --fusion <d1+d2+...>  Fusion components (e.g., hardnet+sosnet)\n";
    std::cout << "  --method <name>       Fusion method: concatenate, average, weighted_avg, max, min\n";
    std::cout << "  --device <dev>        Device: auto, cpu, cuda (default: auto)\n";
    std::cout << "  --no-easy             Skip easy patches\n";
    std::cout << "  --no-hard             Skip hard patches\n";
    std::cout << "  --no-tough            Skip tough patches\n";
    std::cout << "  --quiet               Suppress progress output\n";
    std::cout << "  --config <file>       YAML config file (overrides other options)\n";
    std::cout << "  --help                Show this help message\n\n";
    std::cout << "Supported Descriptors:\n";
    for (const auto& name : thesis_project::patches::PatchDescriptorFactory::supportedTypes()) {
        std::cout << "  - " << name << "\n";
    }
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --descriptor hardnet\n";
    std::cout << "  " << prog << " --fusion \"hardnet+sosnet\" --method concat\n";
    std::cout << "  " << prog << " --fusion \"sift+hardnet\" --method average\n";
}

Args parseArgs(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--patches" && i + 1 < argc) {
            args.patches_dir = argv[++i];
        } else if (arg == "--descriptor" && i + 1 < argc) {
            args.descriptor = argv[++i];
        } else if (arg == "--fusion" && i + 1 < argc) {
            std::string fusion_str = argv[++i];
            // Parse "hardnet+sosnet" into components
            size_t pos = 0;
            while ((pos = fusion_str.find('+')) != std::string::npos) {
                args.fusion_components.push_back(fusion_str.substr(0, pos));
                fusion_str.erase(0, pos + 1);
            }
            if (!fusion_str.empty()) {
                args.fusion_components.push_back(fusion_str);
            }
        } else if (arg == "--method" && i + 1 < argc) {
            args.fusion_method = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            args.device = argv[++i];
        } else if (arg == "--no-easy") {
            args.easy = false;
        } else if (arg == "--no-hard") {
            args.hard = false;
        } else if (arg == "--no-tough") {
            args.tough = false;
        } else if (arg == "--quiet") {
            args.verbose = false;
        } else if (arg == "--config" && i + 1 < argc) {
            args.config_file = argv[++i];
        } else if (arg[0] != '-' && args.config_file.empty()) {
            args.config_file = arg;
        }
    }

    return args;
}

std::vector<std::string> resolveScenePaths(const YAML::Node& scenes_node,
                                           const std::string& patches_dir) {
    std::vector<std::string> scenes;
    if (!scenes_node || !scenes_node.IsSequence()) {
        return scenes;
    }

    for (const auto& entry : scenes_node) {
        if (!entry.IsScalar()) {
            throw std::runtime_error("patches.scenes entries must be strings");
        }
        std::string scene = entry.as<std::string>();
        if (scene.empty()) {
            continue;
        }
        std::filesystem::path scene_path(scene);
        if (scene_path.is_absolute()) {
            scenes.push_back(scene_path.string());
        } else {
            scenes.push_back((std::filesystem::path(patches_dir) / scene).string());
        }
    }

    return scenes;
}

std::string joinStrings(const std::vector<std::string>& values, const std::string& delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << delimiter;
        }
        oss << values[i];
    }
    return oss.str();
}

std::string toLowerCopy(const std::string& input) {
    std::string value = input;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

DescriptorConfig parseDescriptorConfig(const YAML::Node& node) {
    DescriptorConfig desc;

    if (node["name"]) {
        desc.name = node["name"].as<std::string>();
    }
    if (node["type"]) {
        desc.type = toLowerCopy(node["type"].as<std::string>());
    }
    if (node["device"]) {
        desc.device = node["device"].as<std::string>();
    }
    if (node["weights"]) {
        if (!node["weights"].IsSequence()) {
            throw std::runtime_error("descriptor.weights must be a sequence");
        }
        for (const auto& weight : node["weights"]) {
            desc.weights.push_back(weight.as<float>());
        }
    }

    if (node["components"]) {
        if (!node["components"].IsSequence()) {
            throw std::runtime_error("descriptor.components must be a sequence");
        }
        for (const auto& comp : node["components"]) {
            if (comp.IsScalar()) {
                desc.components.push_back(comp.as<std::string>());
            } else if (comp["type"]) {
                desc.components.push_back(comp["type"].as<std::string>());
            } else {
                throw std::runtime_error("descriptor.components entries must be strings or {type: ...}");
            }
        }
    }

    if (node["aggregation"]) {
        desc.method = node["aggregation"].as<std::string>();
    } else if (node["method"]) {
        desc.method = node["method"].as<std::string>();
    } else if (node["fusion_method"]) {
        desc.method = node["fusion_method"].as<std::string>();
    }

    desc.method = toLowerCopy(desc.method);

    if (desc.isFusion() && desc.components.empty()) {
        throw std::runtime_error("composite descriptor requires components");
    }
    if (!desc.isFusion() && desc.type.empty()) {
        throw std::runtime_error("descriptor type is required");
    }
    if (desc.type.empty() && !desc.components.empty()) {
        desc.type = "composite";
    }

    return desc;
}

BenchmarkConfig loadConfig(const std::string& path) {
    BenchmarkConfig config;
    config.benchmark.patches_dir = "../hpatches-release";

    YAML::Node root = YAML::LoadFile(path);

    if (root["patches"]) {
        const auto& patches = root["patches"];
        if (patches["path"]) {
            config.benchmark.patches_dir = patches["path"].as<std::string>();
        }
        if (patches["scenes"]) {
            config.benchmark.scenes = resolveScenePaths(patches["scenes"], config.benchmark.patches_dir);
        }
        if (patches["difficulty"]) {
            const auto& difficulty = patches["difficulty"];
            if (difficulty["easy"]) config.benchmark.include_easy = difficulty["easy"].as<bool>();
            if (difficulty["hard"]) config.benchmark.include_hard = difficulty["hard"].as<bool>();
            if (difficulty["tough"]) config.benchmark.include_tough = difficulty["tough"].as<bool>();
        }
    }

    if (root["performance"]) {
        const auto& performance = root["performance"];
        if (performance["num_threads"]) {
            config.benchmark.num_threads = performance["num_threads"].as<int>();
        }
        if (performance["verbose"]) {
            config.benchmark.verbose = performance["verbose"].as<bool>();
        }
    }

    if (root["output"]) {
        const auto& output = root["output"];
        if (output["print_results"]) {
            config.benchmark.print_results = output["print_results"].as<bool>();
        }
        if (output["save_to_database"]) {
            config.save_to_database = output["save_to_database"].as<bool>();
        }
    }

    if (root["descriptors"]) {
        if (!root["descriptors"].IsSequence()) {
            throw std::runtime_error("descriptors must be a list");
        }
        for (const auto& node : root["descriptors"]) {
            config.descriptors.push_back(parseDescriptorConfig(node));
        }
    }

    return config;
}

void printSummary(const std::vector<thesis_project::benchmark::HPatchesBenchmark::Results>& results) {
    if (results.empty()) {
        return;
    }

    std::cout << "\nSummary (mAP overall)\n";
    std::cout << "---------------------\n";
    std::cout << std::left << std::setw(32) << "Descriptor"
              << std::right << std::setw(12) << "mAP" << "\n";
    std::cout << "---------------------\n";

    for (const auto& res : results) {
        std::ostringstream map_str;
        map_str << std::fixed << std::setprecision(1) << (res.mAP_overall * 100.0f) << "%";
        std::cout << std::left << std::setw(32) << res.descriptor_name
                  << std::right << std::setw(12) << map_str.str() << "\n";
    }
    std::cout << "---------------------\n";
}

int main(int argc, char* argv[]) {
    using namespace thesis_project;
    using namespace thesis_project::patches;
    using namespace thesis_project::benchmark;
    using thesis_project::database::DatabaseManager;
    using thesis_project::database::ExperimentConfig;
    using thesis_project::database::PatchBenchmarkResults;

    Args args = parseArgs(argc, argv);

    if (args.help) {
        printUsage(argv[0]);
        return 0;
    }

    try {
        if (!args.config_file.empty()) {
            BenchmarkConfig config = loadConfig(args.config_file);

            if (config.descriptors.empty()) {
                throw std::runtime_error("No descriptors defined in config file");
            }

            std::unique_ptr<DatabaseManager> db;
            if (config.save_to_database) {
                db = std::make_unique<DatabaseManager>("experiments.db", true);
                if (!db->isEnabled()) {
                    std::cerr << "Warning: Failed to connect to database, disabling save_to_database\n";
                    db.reset();
                }
            }

            std::vector<HPatchesBenchmark::Results> all_results;
            all_results.reserve(config.descriptors.size());

            int descriptor_index = 0;
            for (const auto& desc_config : config.descriptors) {
                descriptor_index++;

                if (config.benchmark.verbose) {
                    std::cout << "\n[" << descriptor_index << "/" << config.descriptors.size()
                              << "] Descriptor: " << (desc_config.name.empty() ? desc_config.type : desc_config.name)
                              << "\n";
                }

                std::unique_ptr<IPatchDescriptorExtractor> extractor;
                if (desc_config.isFusion()) {
                    extractor = PatchDescriptorFactory::createFusion(
                        desc_config.components,
                        desc_config.method,
                        desc_config.weights,
                        desc_config.name);
                } else {
                    extractor = PatchDescriptorFactory::create(desc_config.type);
                    if (!desc_config.name.empty()) {
                        extractor = std::make_unique<NamedPatchDescriptorExtractor>(
                            std::move(extractor), desc_config.name);
                    }
                }

                DescriptorParams params;
                params.device = desc_config.device;

                auto results = HPatchesBenchmark::run(
                    config.benchmark,
                    *extractor,
                    params,
                    [&config](int current, int total, const std::string& scene) {
                        if (config.benchmark.verbose) {
                            std::cout << "\rProcessing scene " << current << "/" << total
                                      << ": " << scene << std::flush;
                        }
                    });

                if (config.benchmark.verbose) {
                    std::cout << "\n";
                }

                if (db) {
                    ExperimentConfig exp_config;
                    exp_config.descriptor_type = desc_config.name.empty()
                        ? extractor->name()
                        : desc_config.name;
                    exp_config.dataset_path = config.benchmark.patches_dir;
                    exp_config.pooling_strategy = "patch_benchmark";
                    exp_config.similarity_threshold = 0.0;
                    exp_config.max_features = 0;
                    exp_config.descriptor_dimension = results.descriptor_dimension;
                    exp_config.execution_device = desc_config.device;
                    exp_config.parameters["benchmark"] = "patch_benchmark";
                    exp_config.parameters["patches_dir"] = config.benchmark.patches_dir;
                    exp_config.parameters["difficulty_easy"] = config.benchmark.include_easy ? "true" : "false";
                    exp_config.parameters["difficulty_hard"] = config.benchmark.include_hard ? "true" : "false";
                    exp_config.parameters["difficulty_tough"] = config.benchmark.include_tough ? "true" : "false";
                    if (!config.benchmark.scenes.empty()) {
                        exp_config.parameters["scenes"] = joinStrings(config.benchmark.scenes, ",");
                    }
                    if (desc_config.isFusion()) {
                        exp_config.parameters["fusion_method"] = desc_config.method;
                        exp_config.parameters["components"] = joinStrings(desc_config.components, "+");
                        if (!desc_config.weights.empty()) {
                            std::vector<std::string> weight_strings;
                            weight_strings.reserve(desc_config.weights.size());
                            for (float weight : desc_config.weights) {
                                weight_strings.push_back(std::to_string(weight));
                            }
                            exp_config.parameters["weights"] = joinStrings(weight_strings, ",");
                        }
                    }
                    exp_config.parameters["config_file"] = args.config_file;

                    const int experiment_id = db->recordConfiguration(exp_config);
                    if (experiment_id >= 0) {
                        PatchBenchmarkResults patch_results;
                        patch_results.experiment_id = experiment_id;
                        patch_results.descriptor_name = results.descriptor_name;
                        patch_results.descriptor_dimension = results.descriptor_dimension;
                        patch_results.map_overall = results.mAP_overall;
                        patch_results.accuracy_overall = results.accuracy_overall;
                        patch_results.map_easy = results.mAP_easy;
                        patch_results.map_hard = results.mAP_hard;
                        patch_results.map_tough = results.mAP_tough;
                        patch_results.map_illumination = results.mAP_illumination;
                        patch_results.map_viewpoint = results.mAP_viewpoint;
                        patch_results.map_illumination_easy = results.mAP_illumination_easy;
                        patch_results.map_illumination_hard = results.mAP_illumination_hard;
                        patch_results.map_viewpoint_easy = results.mAP_viewpoint_easy;
                        patch_results.map_viewpoint_hard = results.mAP_viewpoint_hard;
                        patch_results.num_scenes = results.num_scenes;
                        patch_results.num_patches = results.num_patches;
                        patch_results.processing_time_ms = results.processing_time_ms;
                        patch_results.metadata["patches_dir"] = config.benchmark.patches_dir;
                        patch_results.metadata["difficulty_easy"] = config.benchmark.include_easy ? "true" : "false";
                        patch_results.metadata["difficulty_hard"] = config.benchmark.include_hard ? "true" : "false";
                        patch_results.metadata["difficulty_tough"] = config.benchmark.include_tough ? "true" : "false";
                        if (!config.benchmark.scenes.empty()) {
                            patch_results.metadata["scenes"] = joinStrings(config.benchmark.scenes, ",");
                        }
                        patch_results.metadata["descriptor_type"] = desc_config.type;
                        patch_results.metadata["execution_device"] = desc_config.device;
                        if (desc_config.isFusion()) {
                            patch_results.metadata["fusion_method"] = desc_config.method;
                            patch_results.metadata["components"] = joinStrings(desc_config.components, "+");
                            if (!desc_config.weights.empty()) {
                                std::vector<std::string> weight_strings;
                                weight_strings.reserve(desc_config.weights.size());
                                for (float weight : desc_config.weights) {
                                    weight_strings.push_back(std::to_string(weight));
                                }
                                patch_results.metadata["weights"] = joinStrings(weight_strings, ",");
                            }
                        }
                        patch_results.metadata["config_file"] = args.config_file;
                        db->recordPatchBenchmarkResults(patch_results);
                    }
                }

                all_results.push_back(results);
            }

            if (config.benchmark.print_results && all_results.size() > 1) {
                printSummary(all_results);
            }

            return 0;
        }

        // Validate arguments
        if (args.descriptor.empty() && args.fusion_components.empty()) {
            std::cerr << "Error: Must specify --descriptor or --fusion\n";
            printUsage(argv[0]);
            return 1;
        }

        // Create the descriptor extractor
        std::unique_ptr<IPatchDescriptorExtractor> extractor;

        if (!args.fusion_components.empty()) {
            // Create fusion extractor
            extractor = PatchDescriptorFactory::createFusion(
                args.fusion_components,
                args.fusion_method);
            std::cout << "Created fusion descriptor: " << extractor->name()
                      << " (" << extractor->descriptorSize() << "D)\n";
        } else {
            // Create single descriptor
            extractor = PatchDescriptorFactory::create(args.descriptor);
            std::cout << "Created descriptor: " << extractor->name()
                      << " (" << extractor->descriptorSize() << "D)\n";
        }

        // Set up benchmark config
        HPatchesBenchmark::Config config;
        config.patches_dir = args.patches_dir;
        config.include_easy = args.easy;
        config.include_hard = args.hard;
        config.include_tough = args.tough;
        config.verbose = args.verbose;
        config.print_results = args.verbose;

        // Set up descriptor params
        DescriptorParams params;
        params.device = args.device;

        // Run the benchmark
        auto results = HPatchesBenchmark::run(
            config,
            *extractor,
            params,
            [&args](int current, int total, const std::string& scene) {
                if (args.verbose) {
                    std::cout << "\rProcessing scene " << current << "/" << total
                              << ": " << scene << std::flush;
                }
            });

        if (args.verbose) {
            std::cout << "\n";  // Clear progress line
        }

        // Results are already printed by HPatchesBenchmark::run if verbose

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
