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
#include "core/benchmark/PatchBenchmark.hpp"
#include "core/benchmark/BenchmarkTypes.hpp"
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
    bool color = false;
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
    bool use_color = false;
    bool use_color_specified = false;
    bool scales_specified = false;
    bool normalize_before_fusion = false;
    thesis_project::DescriptorParams params;

    bool isFusion() const {
        return !components.empty() || type == "composite" || type == "fusion";
    }
};

struct BenchmarkConfig {
    thesis_project::benchmark::Config benchmark;
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
    std::cout << "  --color               Load color patches (3-channel)\n";
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
        } else if (arg == "--color") {
            args.color = true;
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

std::string serializeParams(const std::map<std::string, std::string>& params) {
    std::ostringstream oss;
    for (const auto& [key, value] : params) {
        oss << key << "=" << value << ";";
    }
    return oss.str();
}

std::string toLowerCopy(const std::string& input) {
    std::string value = input;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : line) {
        if (c == ',') {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    parts.push_back(current);
    return parts;
}

std::vector<thesis_project::database::DatabaseManager::PatchBenchmarkTaskPair>
loadVerificationPairsCsv(const std::string& path) {
    std::vector<thesis_project::database::DatabaseManager::PatchBenchmarkTaskPair> pairs;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open verification tasks file: " + path);
    }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            continue;
        }
        auto cols = splitCsvLine(line);
        if (cols.size() < 6) {
            continue;
        }
        thesis_project::database::DatabaseManager::PatchBenchmarkTaskPair pair;
        pair.s1 = cols[0];
        pair.t1 = std::stoi(cols[1]);
        pair.idx1 = std::stoi(cols[2]);
        pair.s2 = cols[3];
        pair.t2 = std::stoi(cols[4]);
        pair.idx2 = std::stoi(cols[5]);
        pairs.push_back(std::move(pair));
    }
    return pairs;
}

std::vector<thesis_project::database::DatabaseManager::PatchBenchmarkTaskItem>
loadRetrievalItemsCsv(const std::string& path) {
    std::vector<thesis_project::database::DatabaseManager::PatchBenchmarkTaskItem> items;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open retrieval tasks file: " + path);
    }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            continue;
        }
        auto cols = splitCsvLine(line);
        if (cols.size() < 2) {
            continue;
        }
        thesis_project::database::DatabaseManager::PatchBenchmarkTaskItem item;
        item.s = cols[0];
        item.idx = std::stoi(cols[1]);
        items.push_back(std::move(item));
    }
    return items;
}

void importTasksToDatabase(thesis_project::database::DatabaseManager& db,
                           const std::string& task_set_name,
                           const std::string& tasks_dir,
                           const std::string& notes) {
    const int task_set_id = db.upsertPatchBenchmarkTaskSet(task_set_name, tasks_dir, notes);
    if (task_set_id < 0) {
        throw std::runtime_error("Failed to create or load task set: " + task_set_name);
    }

    const std::vector<std::string> splits = {"full", "illum", "view", "a", "b", "c"};
    for (const auto& split : splits) {
        const std::string pos_path = (std::filesystem::path(tasks_dir) / ("verif_pos_split-" + split + ".csv")).string();
        if (std::filesystem::exists(pos_path)) {
            db.storePatchBenchmarkVerificationPairs(task_set_id, split, "pos", loadVerificationPairsCsv(pos_path));
        }
        const std::string neg_inter = (std::filesystem::path(tasks_dir) / ("verif_neg_inter_split-" + split + ".csv")).string();
        if (std::filesystem::exists(neg_inter)) {
            db.storePatchBenchmarkVerificationPairs(task_set_id, split, "inter", loadVerificationPairsCsv(neg_inter));
        }
        const std::string neg_intra = (std::filesystem::path(tasks_dir) / ("verif_neg_intra_split-" + split + ".csv")).string();
        if (std::filesystem::exists(neg_intra)) {
            db.storePatchBenchmarkVerificationPairs(task_set_id, split, "intra", loadVerificationPairsCsv(neg_intra));
        }
        const std::string retr_queries = (std::filesystem::path(tasks_dir) / ("retr_queries_split-" + split + ".csv")).string();
        if (std::filesystem::exists(retr_queries)) {
            db.storePatchBenchmarkRetrievalQueries(task_set_id, split, loadRetrievalItemsCsv(retr_queries));
        }
        const std::string retr_distractors = (std::filesystem::path(tasks_dir) / ("retr_distractors_split-" + split + ".csv")).string();
        if (std::filesystem::exists(retr_distractors)) {
            db.storePatchBenchmarkRetrievalDistractors(task_set_id, split, loadRetrievalItemsCsv(retr_distractors));
        }
    }
}

void loadTasksFromDatabase(thesis_project::database::DatabaseManager& db,
                           const std::string& task_set_name,
                           const std::string& split,
                           thesis_project::benchmark::Config& config) {
    const int task_set_id = db.getPatchBenchmarkTaskSetId(task_set_name);
    if (task_set_id < 0) {
        throw std::runtime_error("Task set not found in database: " + task_set_name);
    }

    const auto pos_pairs_db = db.loadPatchBenchmarkVerificationPairs(task_set_id, split, "pos");
    const auto neg_inter_db = db.loadPatchBenchmarkVerificationPairs(task_set_id, split, "inter");
    const auto neg_intra_db = db.loadPatchBenchmarkVerificationPairs(task_set_id, split, "intra");
    const auto queries_db = db.loadPatchBenchmarkRetrievalQueries(task_set_id, split);
    const auto distractors_db = db.loadPatchBenchmarkRetrievalDistractors(task_set_id, split);

    config.verification_pos_pairs.clear();
    config.verification_pos_pairs.reserve(pos_pairs_db.size());
    for (const auto& p : pos_pairs_db) {
        config.verification_pos_pairs.push_back({p.s1, p.t1, p.idx1, p.s2, p.t2, p.idx2});
    }

    config.verification_neg_inter_pairs.clear();
    config.verification_neg_inter_pairs.reserve(neg_inter_db.size());
    for (const auto& p : neg_inter_db) {
        config.verification_neg_inter_pairs.push_back({p.s1, p.t1, p.idx1, p.s2, p.t2, p.idx2});
    }

    config.verification_neg_intra_pairs.clear();
    config.verification_neg_intra_pairs.reserve(neg_intra_db.size());
    for (const auto& p : neg_intra_db) {
        config.verification_neg_intra_pairs.push_back({p.s1, p.t1, p.idx1, p.s2, p.t2, p.idx2});
    }

    config.retrieval_queries.clear();
    config.retrieval_queries.reserve(queries_db.size());
    for (const auto& q : queries_db) {
        config.retrieval_queries.push_back({q.s, q.idx});
    }

    config.retrieval_distractors.clear();
    config.retrieval_distractors.reserve(distractors_db.size());
    for (const auto& d : distractors_db) {
        config.retrieval_distractors.push_back({d.s, d.idx});
    }
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
        desc.params.device = desc.device;
    }

    if (node["use_color"]) {
        desc.use_color = node["use_color"].as<bool>();
        desc.use_color_specified = true;
        desc.params.use_color = desc.use_color;
    }

    if (node["scales"]) {
        if (!node["scales"].IsSequence()) {
            throw std::runtime_error("descriptor.scales must be a list");
        }
        desc.params.scales.clear();
        for (const auto& scale_node : node["scales"]) {
            desc.params.scales.push_back(scale_node.as<float>());
        }
        desc.scales_specified = true;
    }

    if (node["scale_weights"]) {
        if (!node["scale_weights"].IsSequence()) {
            throw std::runtime_error("descriptor.scale_weights must be a list");
        }
        desc.params.scale_weights.clear();
        for (const auto& weight_node : node["scale_weights"]) {
            desc.params.scale_weights.push_back(weight_node.as<float>());
        }
    }

    if (node["scale_weighting"]) {
        const std::string wt = toLowerCopy(node["scale_weighting"].as<std::string>());
        if (wt == "gaussian") desc.params.scale_weighting = thesis_project::ScaleWeighting::GAUSSIAN;
        else if (wt == "triangular") desc.params.scale_weighting = thesis_project::ScaleWeighting::TRIANGULAR;
        else desc.params.scale_weighting = thesis_project::ScaleWeighting::UNIFORM;
    }

    if (node["scale_weight_sigma"]) {
        desc.params.scale_weight_sigma = node["scale_weight_sigma"].as<float>();
    }

    if (node["pooling_aggregation"]) {
        const std::string agg = toLowerCopy(node["pooling_aggregation"].as<std::string>());
        if (agg == "max") desc.params.pooling_aggregation = thesis_project::PoolingAggregation::MAX;
        else if (agg == "min") desc.params.pooling_aggregation = thesis_project::PoolingAggregation::MIN;
        else if (agg == "concatenate") desc.params.pooling_aggregation = thesis_project::PoolingAggregation::CONCATENATE;
        else if (agg == "weighted_avg") desc.params.pooling_aggregation = thesis_project::PoolingAggregation::WEIGHTED_AVG;
        else desc.params.pooling_aggregation = thesis_project::PoolingAggregation::AVERAGE;
    }

    if (node["normalize_before_pooling"]) {
        desc.params.normalize_before_pooling = node["normalize_before_pooling"].as<bool>();
    }

    if (node["normalize_after_pooling"]) {
        desc.params.normalize_after_pooling = node["normalize_after_pooling"].as<bool>();
    }

    if (node["rooting_stage"]) {
        const std::string stage = toLowerCopy(node["rooting_stage"].as<std::string>());
        if (stage == "before_pooling") desc.params.rooting_stage = thesis_project::RootingStage::R_BEFORE_POOLING;
        else if (stage == "after_pooling") desc.params.rooting_stage = thesis_project::RootingStage::R_AFTER_POOLING;
        else desc.params.rooting_stage = thesis_project::RootingStage::R_NONE;
    }

    if (node["norm_type"]) {
        const std::string norm_str = toLowerCopy(node["norm_type"].as<std::string>());
        if (norm_str == "l1") desc.params.norm_type = cv::NORM_L1;
        else desc.params.norm_type = cv::NORM_L2;
    }

    if (node["patch_keypoint_size"]) {
        desc.params.patch_keypoint_size = node["patch_keypoint_size"].as<float>();
    } else if (node["keypoint_size"]) {
        desc.params.patch_keypoint_size = node["keypoint_size"].as<float>();
    }

    if (node["extended"]) {
        desc.params.surf_extended = node["extended"].as<bool>();
    }

    if (node["dnn"]) {
        const auto& dnn = node["dnn"];
        if (dnn["model"]) desc.params.dnn_model_path = dnn["model"].as<std::string>();
        if (dnn["input_size"]) desc.params.dnn_input_size = dnn["input_size"].as<int>();
        if (dnn["support_multiplier"]) desc.params.dnn_support_multiplier = dnn["support_multiplier"].as<float>();
        if (dnn["rotate_to_upright"]) desc.params.dnn_rotate_upright = dnn["rotate_to_upright"].as<bool>();
        if (dnn["mean"]) desc.params.dnn_mean = dnn["mean"].as<float>();
        if (dnn["std"]) desc.params.dnn_std = dnn["std"].as<float>();
        if (dnn["per_patch_standardize"]) desc.params.dnn_per_patch_standardize = dnn["per_patch_standardize"].as<bool>();
    }

    if (node["vgg"]) {
        const auto& vgg = node["vgg"];
        if (vgg["desc_type"]) desc.params.vgg_desc_type = vgg["desc_type"].as<int>();
        if (vgg["isigma"]) desc.params.vgg_isigma = vgg["isigma"].as<float>();
        if (vgg["img_normalize"]) desc.params.vgg_img_normalize = vgg["img_normalize"].as<bool>();
        if (vgg["use_scale_orientation"]) desc.params.vgg_use_scale_orientation = vgg["use_scale_orientation"].as<bool>();
        if (vgg["scale_factor"]) desc.params.vgg_scale_factor = vgg["scale_factor"].as<float>();
        if (vgg["dsc_normalize"]) desc.params.vgg_dsc_normalize = vgg["dsc_normalize"].as<bool>();
    }
    if (node["normalize_before_fusion"]) {
        desc.normalize_before_fusion = node["normalize_before_fusion"].as<bool>();
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
        if (patches["color"]) {
            config.benchmark.color = patches["color"].as<bool>();
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

    if (root["tasks"]) {
        const auto& tasks = root["tasks"];
        auto parseEnabled = [](const YAML::Node& node, bool default_value) {
            if (!node) return default_value;
            if (node.IsScalar()) return node.as<bool>();
            if (node["enabled"]) return node["enabled"].as<bool>();
            return default_value;
        };

        if (tasks["task_source"]) {
            config.benchmark.task_source = toLowerCopy(tasks["task_source"].as<std::string>());
        }
        if (tasks["task_set"]) {
            config.benchmark.task_set = tasks["task_set"].as<std::string>();
        }
        if (tasks["task_split"]) {
            config.benchmark.task_split = toLowerCopy(tasks["task_split"].as<std::string>());
        }
        if (tasks["tasks_dir"]) {
            config.benchmark.tasks_dir = tasks["tasks_dir"].as<std::string>();
        }
        if (tasks["matching"]) {
            config.benchmark.matching_enabled = parseEnabled(tasks["matching"], config.benchmark.matching_enabled);
        }
        if (tasks["verification"]) {
            const auto& ver = tasks["verification"];
            config.benchmark.verification_enabled = parseEnabled(ver, config.benchmark.verification_enabled);
            if (ver["num_positives"]) {
                config.benchmark.verification_num_positives = ver["num_positives"].as<int>();
            }
            if (ver["num_negatives"]) {
                config.benchmark.verification_num_negatives = ver["num_negatives"].as<int>();
            }
            if (ver["negative_source"]) {
                const std::string source = toLowerCopy(ver["negative_source"].as<std::string>());
                if (source == "same_seq") {
                    config.benchmark.verification_same_seq = true;
                    config.benchmark.verification_diff_seq = false;
                } else if (source == "diff_seq") {
                    config.benchmark.verification_same_seq = false;
                    config.benchmark.verification_diff_seq = true;
                } else if (source == "both") {
                    config.benchmark.verification_same_seq = true;
                    config.benchmark.verification_diff_seq = true;
                } else {
                    throw std::runtime_error("tasks.verification.negative_source must be same_seq, diff_seq, or both");
                }
            }
        }
        if (tasks["retrieval"]) {
            const auto& ret = tasks["retrieval"];
            config.benchmark.retrieval_enabled = parseEnabled(ret, config.benchmark.retrieval_enabled);
            if (ret["num_queries"]) {
                config.benchmark.retrieval_num_queries = ret["num_queries"].as<int>();
            }
            if (ret["num_distractors"]) {
                config.benchmark.retrieval_num_distractors = ret["num_distractors"].as<int>();
            }
        }
        if (tasks["store_descriptors_to_db"]) {
            config.benchmark.store_descriptors_to_db = tasks["store_descriptors_to_db"].as<bool>();
        }
        if (tasks["use_cached_descriptors"]) {
            config.benchmark.use_cached_descriptors = tasks["use_cached_descriptors"].as<bool>();
        }
        if (tasks["descriptor_cache_name"]) {
            config.benchmark.descriptor_cache_name = tasks["descriptor_cache_name"].as<std::string>();
        }
        if (tasks["random_seed"]) {
            config.benchmark.random_seed = tasks["random_seed"].as<unsigned int>();
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

void printSummary(const std::vector<thesis_project::benchmark::Results>& results) {
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
            if (!db && (config.benchmark.store_descriptors_to_db ||
                        config.benchmark.use_cached_descriptors)) {
                db = std::make_unique<DatabaseManager>("experiments.db", true);
                if (!db->isEnabled()) {
                    std::cerr << "Warning: Failed to connect to database for descriptor caching\n";
                    db.reset();
                }
            }

            std::unique_ptr<DatabaseManager> task_db;
            DatabaseManager* task_db_ptr = db.get();
            if (config.benchmark.task_source != "random") {
                if (!task_db_ptr) {
                    task_db = std::make_unique<DatabaseManager>("experiments.db", true);
                    if (!task_db->isEnabled()) {
                        throw std::runtime_error("Failed to connect to database for task import/load");
                    }
                    task_db_ptr = task_db.get();
                }

                const std::string source = toLowerCopy(config.benchmark.task_source);
                const std::string split = config.benchmark.task_split.empty()
                    ? "full"
                    : config.benchmark.task_split;
                if (source == "csv") {
                    if (config.benchmark.tasks_dir.empty()) {
                        throw std::runtime_error("tasks.tasks_dir must be set when task_source is csv");
                    }
                    importTasksToDatabase(*task_db_ptr,
                                          config.benchmark.task_set,
                                          config.benchmark.tasks_dir,
                                          "imported from csv");
                    loadTasksFromDatabase(*task_db_ptr, config.benchmark.task_set, split, config.benchmark);
                } else if (source == "db") {
                    loadTasksFromDatabase(*task_db_ptr, config.benchmark.task_set, split, config.benchmark);
                } else if (source != "random") {
                    throw std::runtime_error("tasks.task_source must be random, db, or csv");
                }

                if (config.benchmark.verbose) {
                    std::cout << "[PatchBenchmark] Loaded task set '" << config.benchmark.task_set
                              << "' split '" << split << "'\n"
                              << "  verification_pos_pairs: " << config.benchmark.verification_pos_pairs.size() << "\n"
                              << "  verification_neg_inter_pairs: " << config.benchmark.verification_neg_inter_pairs.size() << "\n"
                              << "  verification_neg_intra_pairs: " << config.benchmark.verification_neg_intra_pairs.size() << "\n"
                              << "  retrieval_queries: " << config.benchmark.retrieval_queries.size() << "\n"
                              << "  retrieval_distractors: " << config.benchmark.retrieval_distractors.size() << "\n";
                }
            }

            std::vector<benchmark::Results> all_results;
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
                        desc_config.name,
                        desc_config.normalize_before_fusion);
                } else {
                    extractor = PatchDescriptorFactory::create(desc_config.type);
                    if (!desc_config.name.empty()) {
                        extractor = std::make_unique<NamedPatchDescriptorExtractor>(
                            std::move(extractor), desc_config.name);
                    }
                }

                DescriptorParams params = desc_config.params;
                if (!desc_config.device.empty()) {
                    params.device = desc_config.device;
                }
                if (desc_config.type == "dspsift_v2" && !desc_config.scales_specified) {
                    // Use DSP defaults (scales + pooling) instead of generic descriptor defaults.
                    params.scales.clear();
                }

                benchmark::Config run_config = config.benchmark;
                if (desc_config.use_color_specified) {
                    run_config.color = desc_config.use_color;
                }

                int experiment_id = -1;
                ExperimentConfig exp_config;
                if (db) {
                    exp_config.descriptor_type = desc_config.name.empty()
                        ? extractor->name()
                        : desc_config.name;
                    exp_config.dataset_path = config.benchmark.patches_dir;
                    exp_config.pooling_strategy = "patch_benchmark";
                    exp_config.similarity_threshold = 0.0;
                    exp_config.max_features = 0;
                    exp_config.descriptor_dimension = extractor->descriptorSize();
                    exp_config.execution_device = desc_config.device;
                    exp_config.parameters["benchmark"] = "patch_benchmark";
                    exp_config.parameters["patches_dir"] = config.benchmark.patches_dir;
                    exp_config.parameters["difficulty_easy"] = config.benchmark.include_easy ? "true" : "false";
                    exp_config.parameters["difficulty_hard"] = config.benchmark.include_hard ? "true" : "false";
                    exp_config.parameters["difficulty_tough"] = config.benchmark.include_tough ? "true" : "false";
                    exp_config.parameters["use_color"] = run_config.color ? "true" : "false";
                    exp_config.parameters["task_matching"] = config.benchmark.matching_enabled ? "true" : "false";
                    exp_config.parameters["task_verification"] = config.benchmark.verification_enabled ? "true" : "false";
                    exp_config.parameters["task_verification_same_seq"] = config.benchmark.verification_same_seq ? "true" : "false";
                    exp_config.parameters["task_verification_diff_seq"] = config.benchmark.verification_diff_seq ? "true" : "false";
                    exp_config.parameters["task_retrieval"] = config.benchmark.retrieval_enabled ? "true" : "false";
                    exp_config.parameters["verification_num_positives"] =
                        std::to_string(config.benchmark.verification_num_positives);
                    exp_config.parameters["verification_num_negatives"] =
                        std::to_string(config.benchmark.verification_num_negatives);
                    exp_config.parameters["retrieval_num_queries"] =
                        std::to_string(config.benchmark.retrieval_num_queries);
                    exp_config.parameters["retrieval_num_distractors"] =
                        std::to_string(config.benchmark.retrieval_num_distractors);
                    exp_config.parameters["task_random_seed"] =
                        std::to_string(config.benchmark.random_seed);
                    exp_config.parameters["task_source"] = config.benchmark.task_source;
                    exp_config.parameters["task_set"] = config.benchmark.task_set;
                    exp_config.parameters["task_split"] = config.benchmark.task_split;
                    exp_config.parameters["descriptor_cache_name"] = config.benchmark.descriptor_cache_name;
                    exp_config.parameters["store_descriptors_to_db"] =
                        config.benchmark.store_descriptors_to_db ? "true" : "false";
                    exp_config.parameters["use_cached_descriptors"] =
                        config.benchmark.use_cached_descriptors ? "true" : "false";
                    exp_config.parameters["patch_keypoint_size"] =
                        std::to_string(params.patch_keypoint_size);
                    if (!config.benchmark.scenes.empty()) {
                        exp_config.parameters["scenes"] = joinStrings(config.benchmark.scenes, ",");
                    }
                    if (desc_config.isFusion()) {
                        exp_config.parameters["fusion_method"] = desc_config.method;
                        exp_config.parameters["components"] = joinStrings(desc_config.components, "+");
                        exp_config.parameters["normalize_before_fusion"] =
                            desc_config.normalize_before_fusion ? "true" : "false";
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

                }

                if (db && (run_config.store_descriptors_to_db || run_config.use_cached_descriptors)) {
                    std::string cache_name = run_config.descriptor_cache_name;
                    if (cache_name.empty()) {
                        cache_name = (desc_config.name.empty() ? extractor->name() : desc_config.name)
                                     + std::string(run_config.color ? "_color" : "_bw");
                    }
                    run_config.descriptor_cache_name = cache_name;
                    if (db) {
                        exp_config.parameters["descriptor_cache_name"] = cache_name;
                    }
                }

                if (db && config.save_to_database) {
                    experiment_id = db->recordConfiguration(exp_config);
                }

                if (db && (run_config.store_descriptors_to_db || run_config.use_cached_descriptors)) {
                    const std::string params_json = serializeParams(exp_config.parameters);
                    const std::string params_hash = std::to_string(std::hash<std::string>{}(params_json));
                    int cache_id = -1;
                    if (run_config.store_descriptors_to_db) {
                        cache_id = db->upsertPatchBenchmarkDescriptorSet(
                            run_config.descriptor_cache_name,
                            experiment_id,
                            exp_config.descriptor_type,
                            extractor->descriptorSize(),
                            run_config.patches_dir,
                            run_config.color,
                            params.patch_keypoint_size,
                            params_hash,
                            params_json);
                    } else {
                        cache_id = db->getPatchBenchmarkDescriptorSetId(run_config.descriptor_cache_name);
                    }
                    if (cache_id < 0 && run_config.use_cached_descriptors && run_config.verbose) {
                        std::cerr << "[PatchBenchmark] Warning: descriptor cache not found for '"
                                  << run_config.descriptor_cache_name << "', falling back to extraction.\n";
                    }
                    run_config.descriptor_cache_id = cache_id;
                }

                auto results = PatchBenchmark::run(
                    run_config,
                    *extractor,
                    params,
                    db.get(),
                    [&config](int current, int total, const std::string& scene) {
                        if (config.benchmark.verbose) {
                            std::cout << "\rProcessing scene " << current << "/" << total
                                      << ": " << scene << std::flush;
                        }
                    });

                if (config.benchmark.verbose) {
                    std::cout << "\n";
                }

                if (db && config.save_to_database && experiment_id >= 0) {
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
                    patch_results.verification_same_overall = results.verification_same_overall;
                    patch_results.verification_same_easy = results.verification_same_easy;
                    patch_results.verification_same_hard = results.verification_same_hard;
                    patch_results.verification_same_tough = results.verification_same_tough;
                    patch_results.verification_same_illumination = results.verification_same_illumination;
                    patch_results.verification_same_viewpoint = results.verification_same_viewpoint;
                    patch_results.verification_same_illumination_easy = results.verification_same_illumination_easy;
                    patch_results.verification_same_illumination_hard = results.verification_same_illumination_hard;
                    patch_results.verification_same_viewpoint_easy = results.verification_same_viewpoint_easy;
                    patch_results.verification_same_viewpoint_hard = results.verification_same_viewpoint_hard;

                    patch_results.verification_diff_overall = results.verification_diff_overall;
                    patch_results.verification_diff_easy = results.verification_diff_easy;
                    patch_results.verification_diff_hard = results.verification_diff_hard;
                    patch_results.verification_diff_tough = results.verification_diff_tough;
                    patch_results.verification_diff_illumination = results.verification_diff_illumination;
                    patch_results.verification_diff_viewpoint = results.verification_diff_viewpoint;
                    patch_results.verification_diff_illumination_easy = results.verification_diff_illumination_easy;
                    patch_results.verification_diff_illumination_hard = results.verification_diff_illumination_hard;
                    patch_results.verification_diff_viewpoint_easy = results.verification_diff_viewpoint_easy;
                    patch_results.verification_diff_viewpoint_hard = results.verification_diff_viewpoint_hard;

                    patch_results.retrieval_overall = results.retrieval_overall;
                    patch_results.retrieval_easy = results.retrieval_easy;
                    patch_results.retrieval_hard = results.retrieval_hard;
                    patch_results.retrieval_tough = results.retrieval_tough;
                    patch_results.retrieval_illumination = results.retrieval_illumination;
                    patch_results.retrieval_viewpoint = results.retrieval_viewpoint;
                    patch_results.retrieval_illumination_easy = results.retrieval_illumination_easy;
                    patch_results.retrieval_illumination_hard = results.retrieval_illumination_hard;
                    patch_results.retrieval_viewpoint_easy = results.retrieval_viewpoint_easy;
                    patch_results.retrieval_viewpoint_hard = results.retrieval_viewpoint_hard;
                    patch_results.verification_negatives_per_query = config.benchmark.verification_num_negatives;
                    patch_results.retrieval_negatives_per_query = config.benchmark.retrieval_num_distractors;
                    if (config.benchmark.verification_same_seq &&
                        config.benchmark.verification_diff_seq) {
                        patch_results.verification_negative_source = "both";
                    } else if (config.benchmark.verification_diff_seq) {
                        patch_results.verification_negative_source = "diff_seq";
                    } else {
                        patch_results.verification_negative_source = "same_seq";
                    }
                    patch_results.retrieval_negative_source = "diff_seq";
                    patch_results.num_scenes = results.num_scenes;
                    patch_results.num_patches = results.num_patches;
                    patch_results.processing_time_ms = results.processing_time_ms;
                    patch_results.metadata["patches_dir"] = config.benchmark.patches_dir;
                    patch_results.metadata["difficulty_easy"] = config.benchmark.include_easy ? "true" : "false";
                    patch_results.metadata["difficulty_hard"] = config.benchmark.include_hard ? "true" : "false";
                    patch_results.metadata["difficulty_tough"] = config.benchmark.include_tough ? "true" : "false";
                    patch_results.metadata["use_color"] = run_config.color ? "true" : "false";
                    patch_results.metadata["task_matching"] = config.benchmark.matching_enabled ? "true" : "false";
                    patch_results.metadata["task_verification"] = config.benchmark.verification_enabled ? "true" : "false";
                    patch_results.metadata["task_verification_same_seq"] = config.benchmark.verification_same_seq ? "true" : "false";
                    patch_results.metadata["task_verification_diff_seq"] = config.benchmark.verification_diff_seq ? "true" : "false";
                    patch_results.metadata["task_retrieval"] = config.benchmark.retrieval_enabled ? "true" : "false";
                    patch_results.metadata["verification_num_positives"] =
                        std::to_string(config.benchmark.verification_num_positives);
                    patch_results.metadata["verification_num_negatives"] =
                        std::to_string(config.benchmark.verification_num_negatives);
                    patch_results.metadata["retrieval_num_queries"] =
                        std::to_string(config.benchmark.retrieval_num_queries);
                    patch_results.metadata["retrieval_num_distractors"] =
                        std::to_string(config.benchmark.retrieval_num_distractors);
                    patch_results.metadata["task_random_seed"] =
                        std::to_string(config.benchmark.random_seed);
                    patch_results.metadata["task_source"] = config.benchmark.task_source;
                    patch_results.metadata["task_set"] = config.benchmark.task_set;
                    patch_results.metadata["task_split"] = config.benchmark.task_split;
                    patch_results.metadata["descriptor_cache_name"] = run_config.descriptor_cache_name;
                    patch_results.metadata["store_descriptors_to_db"] =
                        run_config.store_descriptors_to_db ? "true" : "false";
                    patch_results.metadata["use_cached_descriptors"] =
                        run_config.use_cached_descriptors ? "true" : "false";
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
        benchmark::Config config;
        config.patches_dir = args.patches_dir;
        config.include_easy = args.easy;
        config.include_hard = args.hard;
        config.include_tough = args.tough;
        config.verbose = args.verbose;
        config.print_results = args.verbose;
        config.color = args.color;

        // Set up descriptor params
        DescriptorParams params;
        params.device = args.device;

        // Run the benchmark
        auto results = PatchBenchmark::run(
            config,
            *extractor,
            params,
            nullptr,
            [&args](int current, int total, const std::string& scene) {
                if (args.verbose) {
                    std::cout << "\rProcessing scene " << current << "/" << total
                              << ": " << scene << std::flush;
                }
            });

        if (args.verbose) {
            std::cout << "\n";  // Clear progress line
        }

        // Results are already printed by PatchBenchmark::run if verbose

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
