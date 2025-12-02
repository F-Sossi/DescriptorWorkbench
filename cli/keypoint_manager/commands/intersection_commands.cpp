#include "intersection_commands.hpp"
#include "thesis_project/logging.hpp"
#include <opencv2/flann.hpp>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace thesis_project::cli::keypoint_commands {

int buildIntersection(thesis_project::database::DatabaseManager& db, int argc, char** argv) {
    std::string source_a_name;
    std::string source_b_name;
    std::string output_a_name;
    std::string output_b_name;
    double tolerance_px = 3.0;
    bool overwrite = false;

    auto usage = [&]() {
        std::cerr << "Usage: " << argv[0]
                  << " build-intersection --source-a <set> --source-b <set> --out-a <set> --out-b <set> [--tolerance px] [--overwrite]"
                  << std::endl;
    };

    int arg_index = 2;
    while (arg_index < argc) {
        std::string arg = argv[arg_index++];
        if (arg == "--source-a") {
            if (arg_index >= argc) {
                std::cerr << "Missing value for --source-a" << std::endl;
                return 1;
            }
            source_a_name = argv[arg_index++];
        } else if (arg == "--source-b") {
            if (arg_index >= argc) {
                std::cerr << "Missing value for --source-b" << std::endl;
                return 1;
            }
            source_b_name = argv[arg_index++];
        } else if (arg == "--out-a") {
            if (arg_index >= argc) {
                std::cerr << "Missing value for --out-a" << std::endl;
                return 1;
            }
            output_a_name = argv[arg_index++];
        } else if (arg == "--out-b") {
            if (arg_index >= argc) {
                std::cerr << "Missing value for --out-b" << std::endl;
                return 1;
            }
            output_b_name = argv[arg_index++];
        } else if (arg == "--tolerance") {
            if (arg_index >= argc) {
                std::cerr << "Missing value for --tolerance" << std::endl;
                return 1;
            }
            tolerance_px = std::stod(argv[arg_index++]);
        } else if (arg == "--overwrite") {
            overwrite = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            usage();
            return 1;
        }
    }

    if (source_a_name.empty() || source_b_name.empty() || output_a_name.empty() || output_b_name.empty()) {
        usage();
        return 1;
    }

    if (tolerance_px <= 0.0) {
        std::cerr << "Tolerance must be positive" << std::endl;
        return 1;
    }

    auto source_a_info = db.getKeypointSetInfo(source_a_name);
    if (!source_a_info) {
        std::cerr << "Source keypoint set not found: " << source_a_name << std::endl;
        return 1;
    }

    auto source_b_info = db.getKeypointSetInfo(source_b_name);
    if (!source_b_info) {
        std::cerr << "Source keypoint set not found: " << source_b_name << std::endl;
        return 1;
    }

    if (!source_a_info->dataset_path.empty() && !source_b_info->dataset_path.empty() &&
        source_a_info->dataset_path != source_b_info->dataset_path) {
        LOG_WARNING("Source sets reference different datasets: " + source_a_info->dataset_path + " vs " + source_b_info->dataset_path);
    }

    auto prepareOutputSet = [&](const thesis_project::database::DatabaseManager::KeypointSetInfo& source_info,
                                const std::string& output_name,
                                const std::string& partner_name,
                                int partner_set_id) -> std::optional<int> {
        int existing_id = db.getKeypointSetId(output_name);

        std::ostringstream desc;
        desc << "Paired subset from " << source_info.name << " matched with " << partner_name
             << " (" << tolerance_px << "px tolerance)";

        if (existing_id >= 0) {
            if (!overwrite) {
                std::cerr << "Output set already exists: " << output_name << " (use --overwrite to replace)" << std::endl;
                return std::nullopt;
            }
            if (!db.clearAllDetectorAttributesForSet(existing_id)) {
                LOG_WARNING("Failed to clear detector attributes for " + output_name + ", proceeding with keypoint overwrite");
            }
            if (!db.clearKeypointsForSet(existing_id)) {
                std::cerr << "Failed to clear existing keypoints for set: " << output_name << std::endl;
                return std::nullopt;
            }
            if (!db.updateIntersectionKeypointSet(existing_id,
                                                   source_info.generator_type,
                                                   source_info.generation_method,
                                                   source_info.max_features,
                                                   source_info.dataset_path,
                                                   desc.str(),
                                                   source_info.boundary_filter_px,
                                                   source_info.id,
                                                   partner_set_id,
                                                   static_cast<float>(tolerance_px),
                                                   "mutual_nearest_neighbor")) {
                std::cerr << "Failed to update metadata for existing intersection set: " << output_name << std::endl;
                return std::nullopt;
            }
            return existing_id;
        }

        int new_id = db.createIntersectionKeypointSet(
            output_name,
            source_info.generator_type,
            source_info.generation_method,
            source_info.max_features,
            source_info.dataset_path,
            desc.str(),
            source_info.boundary_filter_px,
            source_info.id,
            partner_set_id,
            static_cast<float>(tolerance_px),
            "mutual_nearest_neighbor"
        );

        if (new_id == -1) {
            std::cerr << "Failed to create output keypoint set: " << output_name << std::endl;
            return std::nullopt;
        }

        return new_id;
    };

    auto output_a_id_opt = prepareOutputSet(*source_a_info, output_a_name, source_b_info->name, source_b_info->id);
    if (!output_a_id_opt) {
        return 1;
    }

    auto output_b_id_opt = prepareOutputSet(*source_b_info, output_b_name, source_a_info->name, source_a_info->id);
    if (!output_b_id_opt) {
        return 1;
    }

    int output_a_id = *output_a_id_opt;
    int output_b_id = *output_b_id_opt;

    auto scenes_a = db.getScenesForSet(source_a_info->id);
    auto scenes_b = db.getScenesForSet(source_b_info->id);

    std::unordered_set<std::string> scene_lookup_b(scenes_b.begin(), scenes_b.end());
    std::vector<std::string> scenes_to_process;
    scenes_to_process.reserve(scenes_a.size());

    for (const auto& scene : scenes_a) {
        if (scene_lookup_b.count(scene)) {
            scenes_to_process.push_back(scene);
        } else {
            LOG_WARNING("Skipping scene " + scene + " (missing in " + source_b_info->name + ")");
        }
    }

    if (scenes_to_process.empty()) {
        std::cerr << "No overlapping scenes between " << source_a_name << " and " << source_b_name << std::endl;
        return 1;
    }

    const float tolerance_sq = static_cast<float>(tolerance_px * tolerance_px);
    cv::flann::SearchParams search_params(32);

    size_t total_pairs = 0;

    for (const auto& scene : scenes_to_process) {
        auto images_a = db.getImagesForSet(source_a_info->id, scene);
        auto images_b = db.getImagesForSet(source_b_info->id, scene);

        std::unordered_set<std::string> images_lookup_b(images_b.begin(), images_b.end());

        for (const auto& image : images_a) {
            if (!images_lookup_b.count(image)) {
                LOG_WARNING("Skipping image " + scene + "/" + image + " (missing in " + source_b_info->name + ")");
                continue;
            }

            auto kp_a_records = db.getLockedKeypointsWithIds(source_a_info->id, scene, image);
            auto kp_b_records = db.getLockedKeypointsWithIds(source_b_info->id, scene, image);

            if (kp_a_records.empty() || kp_b_records.empty()) {
                continue;
            }

            cv::Mat data_b(static_cast<int>(kp_b_records.size()), 2, CV_32F);
            for (size_t i = 0; i < kp_b_records.size(); ++i) {
                data_b.at<float>(static_cast<int>(i), 0) = kp_b_records[i].keypoint.pt.x;
                data_b.at<float>(static_cast<int>(i), 1) = kp_b_records[i].keypoint.pt.y;
            }

            cv::flann::Index kdtree(data_b, cv::flann::KDTreeIndexParams(4));

            std::vector<std::pair<int64_t, cv::KeyPoint>> intersection_a;
            std::vector<std::pair<int64_t, cv::KeyPoint>> intersection_b;

            for (const auto& kp_a : kp_a_records) {
                cv::Mat query(1, 2, CV_32F);
                query.at<float>(0, 0) = kp_a.keypoint.pt.x;
                query.at<float>(0, 1) = kp_a.keypoint.pt.y;

                std::vector<int> indices(1);
                std::vector<float> dists(1);
                kdtree.knnSearch(query, indices, dists, 1, search_params);

                int idx_b = indices[0];
                float dist_sq = dists[0];
                if (dist_sq > tolerance_sq) {
                    continue;
                }

                const auto& kp_b = kp_b_records[static_cast<size_t>(idx_b)];

                // Mutual nearest neighbor check
                cv::Mat query_b(1, 2, CV_32F);
                query_b.at<float>(0, 0) = kp_b.keypoint.pt.x;
                query_b.at<float>(0, 1) = kp_b.keypoint.pt.y;

                std::vector<int> indices_back(1);
                std::vector<float> dists_back(1);
                kdtree.knnSearch(query_b, indices_back, dists_back, 1, search_params);

                if (indices_back[0] != static_cast<int>(&kp_a - &kp_a_records[0])) {
                    continue;
                }

                intersection_a.emplace_back(kp_a.id, kp_a.keypoint);
                intersection_b.emplace_back(kp_b.id, kp_b.keypoint);
            }

            if (!intersection_a.empty()) {
                std::vector<cv::KeyPoint> kps_a;
                std::vector<cv::KeyPoint> kps_b;
                kps_a.reserve(intersection_a.size());
                kps_b.reserve(intersection_b.size());

                for (size_t i = 0; i < intersection_a.size(); ++i) {
                    kps_a.push_back(intersection_a[i].second);
                    kps_b.push_back(intersection_b[i].second);
                }

                db.storeLockedKeypointsForSet(output_a_id, scene, image, kps_a);
                db.storeLockedKeypointsForSet(output_b_id, scene, image, kps_b);

                total_pairs += intersection_a.size();
                LOG_INFO("Stored " + std::to_string(intersection_a.size()) + " intersected pairs for " + scene + "/" + image);
            }
        }
    }

    LOG_INFO("Intersection build complete. Total pairs: " + std::to_string(total_pairs));
    return 0;
}

} // namespace thesis_project::cli::keypoint_commands
