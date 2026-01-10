#include "subset_commands.hpp"
#include "thesis_project/logging.hpp"
#include <algorithm>
#include <chrono>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace thesis_project::cli::keypoint_commands {

int generateRandomSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " generate-random-subset <source_set> <target_count> <output_name> [--seed N]" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-random-subset sift_8000 645448 sift_random_645k --seed 42" << std::endl;
        return 1;
    }

    std::string source_set_name = argv[2];
    size_t target_count = std::stoull(argv[3]);
    std::string output_set_name = argv[4];
    uint32_t seed = 42; // Default seed

    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        }
    }

    LOG_INFO("Generating random subset from: " + source_set_name);
    LOG_INFO("Target count: " + std::to_string(target_count));
    LOG_INFO("Output set: " + output_set_name);
    LOG_INFO("Random seed: " + std::to_string(seed));

    auto source_info = db.getKeypointSetInfo(source_set_name);
    if (!source_info) {
        std::cerr << "Source keypoint set not found: " + source_set_name << std::endl;
        return 1;
    }

    if (db.getKeypointSetInfo(output_set_name)) {
        std::cerr << "Output keypoint set already exists: " + output_set_name << std::endl;
        std::cerr << "Use --overwrite if replacement is desired (not yet supported here)" << std::endl;
        return 1;
    }

    auto output_set_id = db.createKeypointSet(
        output_set_name,
        source_info->generator_type,
        "random_subset",
        source_info->max_features,
        source_info->dataset_path,
        "Random subset of " + source_set_name + ", count=" + std::to_string(target_count) + ", seed=" + std::to_string(seed)
    );

    if (!output_set_id) {
        std::cerr << "Failed to create output keypoint set" << std::endl;
        return 1;
    }

    LOG_INFO("Loading keypoint IDs from source set...");
    std::vector<int64_t> all_ids;
    auto scenes = db.getScenesForSet(source_info->id);
    for (const auto& scene : scenes) {
        auto images = db.getImagesForSet(source_info->id, scene);
        for (const auto& image : images) {
            auto records = db.getLockedKeypointsWithIds(source_info->id, scene, image);
            for (const auto& record : records) {
                all_ids.push_back(record.id);
            }
        }
    }

    LOG_INFO("Loaded " + std::to_string(all_ids.size()) + " keypoint IDs");

    if (all_ids.size() < target_count) {
        std::cerr << "Source set has " << all_ids.size()
                  << " keypoints, cannot sample " << target_count << std::endl;
        return 1;
    }

    std::mt19937 rng(seed);
    std::shuffle(all_ids.begin(), all_ids.end(), rng);
    all_ids.resize(target_count);

    LOG_INFO("Storing " + std::to_string(target_count) + " randomly selected keypoints...");

    size_t stored_count = 0;
    for (const auto& scene : scenes) {
        auto images = db.getImagesForSet(source_info->id, scene);
        for (const auto& image : images) {
            auto records = db.getLockedKeypointsWithIds(source_info->id, scene, image);
            std::vector<cv::KeyPoint> selected_keypoints;

            for (const auto& record : records) {
                if (std::find(all_ids.begin(), all_ids.end(), record.id) != all_ids.end()) {
                    selected_keypoints.push_back(record.keypoint);
                }
            }

            if (!selected_keypoints.empty()) {
                db.storeLockedKeypointsForSet(output_set_id, scene, image, selected_keypoints);
                stored_count += selected_keypoints.size();
            }
        }
    }

    LOG_INFO("Random subset generated successfully.");
    LOG_INFO("Output set: " + output_set_name);
    LOG_INFO("Keypoints stored: " + std::to_string(stored_count));
    LOG_INFO("Seed used: " + std::to_string(seed));
    return 0;
}

int generateTopNSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " generate-top-n <source_set> <target_count> --by <property> --out <output_name>" << std::endl;
        std::cerr << "  Properties: response, size, octave" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-top-n sift_8000 645448 --by response --out sift_top_response_645k" << std::endl;
        return 1;
    }

    std::string source_set_name = argv[2];
    size_t target_count = std::stoull(argv[3]);
    std::string property;
    std::string output_set_name;

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--by" && i + 1 < argc) {
            property = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            output_set_name = argv[++i];
        }
    }

    if (property.empty() || output_set_name.empty()) {
        std::cerr << "Missing required arguments: --by and --out are required." << std::endl;
        return 1;
    }

    if (property != "response" && property != "size" && property != "octave") {
        std::cerr << "Invalid property: " << property << " (must be response, size, or octave)" << std::endl;
        return 1;
    }

    LOG_INFO("Generating top-N subset from: " + source_set_name);
    LOG_INFO("Target count: " + std::to_string(target_count));
    LOG_INFO("Sorting by: " + property);
    LOG_INFO("Output set: " + output_set_name);

    auto source_info = db.getKeypointSetInfo(source_set_name);
    if (!source_info) {
        std::cerr << "Source keypoint set not found: " + source_set_name << std::endl;
        return 1;
    }

    if (db.getKeypointSetInfo(output_set_name)) {
        std::cerr << "Output keypoint set already exists: " + output_set_name << std::endl;
        return 1;
    }

    auto output_set_id = db.createKeypointSet(
        output_set_name,
        source_info->generator_type,
        "top_n_" + property,
        source_info->max_features,
        source_info->dataset_path,
        "Top " + std::to_string(target_count) + " keypoints by " + property + " from " + source_set_name
    );

    if (!output_set_id) {
        std::cerr << "Failed to create output keypoint set" << std::endl;
        return 1;
    }

    struct KeypointRecord {
        int64_t id;
        cv::KeyPoint keypoint;
        std::string scene;
        std::string image;
        double sort_value;
    };

    LOG_INFO("Loading all keypoints from source set...");
    std::vector<KeypointRecord> all_keypoints;
    auto scenes = db.getScenesForSet(source_info->id);
    for (const auto& scene : scenes) {
        auto images = db.getImagesForSet(source_info->id, scene);
        for (const auto& image : images) {
            auto records = db.getLockedKeypointsWithIds(source_info->id, scene, image);
            for (const auto& record : records) {
                double sort_val = 0.0;
                if (property == "response") {
                    sort_val = record.keypoint.response;
                } else if (property == "size") {
                    sort_val = record.keypoint.size;
                } else if (property == "octave") {
                    sort_val = static_cast<double>(record.keypoint.octave);
                }
                all_keypoints.push_back({record.id, record.keypoint, scene, image, sort_val});
            }
        }
    }

    LOG_INFO("Loaded " + std::to_string(all_keypoints.size()) + " keypoints");

    if (all_keypoints.size() < target_count) {
        std::cerr << "Source set only has " << all_keypoints.size()
                  << " keypoints, cannot select top " << target_count << std::endl;
        return 1;
    }

    LOG_INFO("Sorting by " + property + "...");
    std::sort(all_keypoints.begin(), all_keypoints.end(),
              [](const KeypointRecord& a, const KeypointRecord& b) {
                  return a.sort_value > b.sort_value;
              });

    all_keypoints.resize(target_count);

    LOG_INFO("Storing top " + std::to_string(target_count) + " keypoints...");
    std::map<std::pair<std::string, std::string>, std::vector<cv::KeyPoint>> grouped;
    for (const auto& rec : all_keypoints) {
        grouped[{rec.scene, rec.image}].push_back(rec.keypoint);
    }

    size_t stored_count = 0;
    for (const auto& [scene_image, keypoints] : grouped) {
        db.storeLockedKeypointsForSet(output_set_id, scene_image.first, scene_image.second, keypoints);
        stored_count += keypoints.size();
    }

    LOG_INFO("Top-N subset generated successfully.");
    LOG_INFO("Output set: " + output_set_name);
    LOG_INFO("Keypoints stored: " + std::to_string(stored_count));
    LOG_INFO("Sorted by: " + property);
    return 0;
}

int generateSpatialSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " generate-spatial-subset <source_set> <target_count> <min_distance> <output_name>" << std::endl;
        std::cerr << "  Example: " << argv[0] << " generate-spatial-subset sift_8000 645448 16.0 sift_spatial_16px_645k" << std::endl;
        return 1;
    }

    std::string source_set_name = argv[2];
    size_t target_count = std::stoull(argv[3]);
    double min_distance = std::stod(argv[4]);
    std::string output_set_name = argv[5];

    LOG_INFO("Generating spatially filtered subset from: " + source_set_name);
    LOG_INFO("Target count: " + std::to_string(target_count));
    LOG_INFO("Minimum distance: " + std::to_string(min_distance) + "px");
    LOG_INFO("Output set: " + output_set_name);

    auto source_info = db.getKeypointSetInfo(source_set_name);
    if (!source_info) {
        std::cerr << "Source keypoint set not found: " + source_set_name << std::endl;
        return 1;
    }

    if (db.getKeypointSetInfo(output_set_name)) {
        std::cerr << "Output keypoint set already exists: " + output_set_name << std::endl;
        return 1;
    }

    auto output_set_id = db.createKeypointSet(
        output_set_name,
        source_info->generator_type,
        "spatial_filter",
        source_info->max_features,
        source_info->dataset_path,
        "Spatially filtered subset, min_dist=" + std::to_string(min_distance) + "px, from " + source_set_name
    );

    if (!output_set_id) {
        std::cerr << "Failed to create output keypoint set" << std::endl;
        return 1;
    }

    size_t total_stored = 0;
    size_t total_processed = 0;
    const double min_dist_sq = min_distance * min_distance;

    auto scenes = db.getScenesForSet(source_info->id);
    for (const auto& scene : scenes) {
        auto images = db.getImagesForSet(source_info->id, scene);
        for (const auto& image : images) {
            auto records = db.getLockedKeypointsWithIds(source_info->id, scene, image);

            std::sort(records.begin(), records.end(),
                      [](const auto& a, const auto& b) {
                          return a.keypoint.response > b.keypoint.response;
                      });

            std::vector<cv::KeyPoint> selected;
            for (const auto& record : records) {
                bool too_close = false;
                for (const auto& sel : selected) {
                    float dx = record.keypoint.pt.x - sel.pt.x;
                    float dy = record.keypoint.pt.y - sel.pt.y;
                    if (dx * dx + dy * dy < min_dist_sq) {
                        too_close = true;
                        break;
                    }
                }

                if (!too_close) {
                    selected.push_back(record.keypoint);
                    if (total_stored + selected.size() >= target_count) {
                        break;
                    }
                }
            }

            if (!selected.empty()) {
                db.storeLockedKeypointsForSet(output_set_id, scene, image, selected);
                total_stored += selected.size();
                total_processed++;
            }

            if (total_stored >= target_count) {
                LOG_WARNING("Reached target count of " + std::to_string(target_count) + " keypoints");
                break;
            }
        }
        if (total_stored >= target_count) break;
    }

    LOG_INFO("Spatial subset generated.");
    LOG_INFO("Output set: " + output_set_name);
    LOG_INFO("Keypoints stored: " + std::to_string(total_stored));
    LOG_INFO("Minimum distance: " + std::to_string(min_distance) + "px");
    LOG_INFO("Images processed: " + std::to_string(total_processed));

    if (total_stored < target_count) {
        LOG_WARNING("Only stored " + std::to_string(total_stored) + " keypoints (target " +
                   std::to_string(target_count) + ")");
        LOG_WARNING("Consider reducing min_distance or using a denser source set");
    }

    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
