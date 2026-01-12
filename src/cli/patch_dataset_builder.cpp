/**
 * @file patch_dataset_builder.cpp
 * @brief Build HPatches-style patch datasets from original images + DB keypoints
 *
 * Builds grayscale and color patch stacks using:
 * - detector keypoints from the database (Harris, Hessian/SURF, DoG/SIFT)
 * - scale multiplier (default 5x)
 * - rotation jitter from existing HPatches metadata (optional)
 */

#include "thesis_project/database/DatabaseManager.hpp"
#include "thesis_project/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kPatchSize = 65;

struct Args {
    std::string images_dir = "../data";
    std::string metadata_dir = "../hpatches-release";
    std::string output_bw = "../hpatches-release-rebuilt-bw";
    std::string output_color = "../hpatches-release-rebuilt-color";
    std::string db_path = "experiments.db";
    std::string set_harris;
    std::string set_hessian;
    std::string set_dog;
    int target_patches = 2000;
    float candidate_multiplier = 1.0f;
    float weight_harris = 0.2f;
    float weight_hessian = 0.4f;
    float weight_dog = 0.4f;
    float scale_multiplier = 5.0f;
    float overlap_threshold = 0.5f;
    bool verbose = true;
};

struct DetectorKeypoint {
    cv::KeyPoint keypoint;
    std::string detector;
};

struct SceneMetadata {
    int target_count = 0;
    std::map<std::string, std::vector<float>> rotjitter;
    std::map<std::string, std::vector<float>> trjitter;
    std::map<std::string, std::vector<float>> scjitter;
    std::map<std::string, std::vector<float>> anisjitter;
};

void printUsage(const char* prog) {
    std::cout << "HPatches Patch Dataset Builder\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " --set-harris <name> --set-hessian <name> --set-dog <name> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --images <dir>         Path to original HPatches images (default: ../data)\n";
    std::cout << "  --metadata <dir>       Path to HPatches patch metadata (rotjitter/overlaps) (default: ../hpatches-release)\n";
    std::cout << "  --output-bw <dir>      Output path for grayscale patch stacks\n";
    std::cout << "  --output-color <dir>   Output path for color patch stacks\n";
    std::cout << "  --db <path>            SQLite database path (default: experiments.db)\n";
    std::cout << "  --set-harris <name>    Keypoint set name for Harris detector\n";
    std::cout << "  --set-hessian <name>   Keypoint set name for Hessian/SURF detector\n";
    std::cout << "  --set-dog <name>       Keypoint set name for DoG/SIFT detector\n";
    std::cout << "  --target <n>           Target patches per scene if metadata missing (default: 2000)\n";
    std::cout << "  --candidate-mult <x>   Multiplier for per-detector candidate pool (default: 1.0)\n";
    std::cout << "  --weight-harris <x>    Quota weight for Harris (default: 0.2)\n";
    std::cout << "  --weight-hessian <x>   Quota weight for Hessian/SURF (default: 0.4)\n";
    std::cout << "  --weight-dog <x>       Quota weight for DoG/SIFT (default: 0.4)\n";
    std::cout << "  --scale-multiplier <x> Patch scale multiplier (default: 5.0)\n";
    std::cout << "  --overlap <x>          Overlap clustering threshold (default: 0.5)\n";
    std::cout << "  --quiet               Suppress progress output\n";
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--images" && i + 1 < argc) {
            args.images_dir = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            args.metadata_dir = argv[++i];
        } else if (arg == "--output-bw" && i + 1 < argc) {
            args.output_bw = argv[++i];
        } else if (arg == "--output-color" && i + 1 < argc) {
            args.output_color = argv[++i];
        } else if (arg == "--db" && i + 1 < argc) {
            args.db_path = argv[++i];
        } else if (arg == "--set-harris" && i + 1 < argc) {
            args.set_harris = argv[++i];
        } else if (arg == "--set-hessian" && i + 1 < argc) {
            args.set_hessian = argv[++i];
        } else if (arg == "--set-dog" && i + 1 < argc) {
            args.set_dog = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            args.target_patches = std::stoi(argv[++i]);
        } else if (arg == "--candidate-mult" && i + 1 < argc) {
            args.candidate_multiplier = std::stof(argv[++i]);
        } else if (arg == "--weight-harris" && i + 1 < argc) {
            args.weight_harris = std::stof(argv[++i]);
        } else if (arg == "--weight-hessian" && i + 1 < argc) {
            args.weight_hessian = std::stof(argv[++i]);
        } else if (arg == "--weight-dog" && i + 1 < argc) {
            args.weight_dog = std::stof(argv[++i]);
        } else if (arg == "--scale-multiplier" && i + 1 < argc) {
            args.scale_multiplier = std::stof(argv[++i]);
        } else if (arg == "--overlap" && i + 1 < argc) {
            args.overlap_threshold = std::stof(argv[++i]);
        } else if (arg == "--quiet") {
            args.verbose = false;
        }
    }
    return args;
}

std::vector<std::string> listScenes(const std::string& root) {
    std::vector<std::string> scenes;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() > 2 && (name[0] == 'i' || name[0] == 'v') && name[1] == '_') {
            scenes.push_back(name);
        }
    }
    std::sort(scenes.begin(), scenes.end());
    return scenes;
}

std::vector<float> loadFloatLines(const fs::path& path) {
    std::vector<float> values;
    std::ifstream file(path);
    if (!file.is_open()) {
        return values;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        values.push_back(std::stof(line));
    }
    return values;
}

SceneMetadata loadSceneMetadata(const fs::path& metadata_scene, int fallback_target) {
    SceneMetadata meta;
    const auto ref_rot = metadata_scene / "ref.rotjitter";
    if (fs::exists(ref_rot)) {
        auto values = loadFloatLines(ref_rot);
        meta.target_count = static_cast<int>(values.size());
    }
    if (meta.target_count <= 0) {
        meta.target_count = fallback_target;
    }

    const std::array<std::string_view, 3> prefixes = {"e", "h", "t"};
    const std::array<std::string_view, 4> suffixes = {"rotjitter", "trjitter", "scjitter", "anisjitter"};
    for (const auto& prefix : prefixes) {
        for (int i = 1; i <= 5; ++i) {
            for (const auto& suffix : suffixes) {
                const std::string key = std::string(prefix) + std::to_string(i);
                const auto file = metadata_scene / (key + "." + std::string(suffix));
                if (!fs::exists(file)) {
                    continue;
                }
                if (suffix == "rotjitter") {
                    meta.rotjitter[key] = loadFloatLines(file);
                } else if (suffix == "trjitter") {
                    meta.trjitter[key] = loadFloatLines(file);
                } else if (suffix == "scjitter") {
                    meta.scjitter[key] = loadFloatLines(file);
                } else if (suffix == "anisjitter") {
                    meta.anisjitter[key] = loadFloatLines(file);
                }
            }
        }
    }

    return meta;
}

double circleOverlapIoU(const cv::KeyPoint& a, const cv::KeyPoint& b) {
    const double r1 = std::max(1e-6, static_cast<double>(a.size) * 0.5);
    const double r2 = std::max(1e-6, static_cast<double>(b.size) * 0.5);
    const double dx = a.pt.x - b.pt.x;
    const double dy = a.pt.y - b.pt.y;
    const double d = std::sqrt(dx * dx + dy * dy);

    if (d >= r1 + r2) {
        return 0.0;
    }
    if (d <= std::abs(r1 - r2)) {
        const double inter = CV_PI * std::min(r1, r2) * std::min(r1, r2);
        const double union_area = CV_PI * r1 * r1 + CV_PI * r2 * r2 - inter;
        return inter / union_area;
    }

    const double alpha = 2.0 * std::acos((r1 * r1 + d * d - r2 * r2) / (2.0 * r1 * d));
    const double beta = 2.0 * std::acos((r2 * r2 + d * d - r1 * r1) / (2.0 * r2 * d));
    const double inter = 0.5 * r1 * r1 * (alpha - std::sin(alpha)) +
                         0.5 * r2 * r2 * (beta - std::sin(beta));
    const double union_area = CV_PI * r1 * r1 + CV_PI * r2 * r2 - inter;
    return inter / union_area;
}

std::vector<DetectorKeypoint> clusterOverlapping(
    const std::vector<DetectorKeypoint>& input,
    float overlap_threshold) {
    std::vector<DetectorKeypoint> result;
    for (const auto& candidate : input) {
        bool overlaps = false;
        for (const auto& kept : result) {
            if (circleOverlapIoU(candidate.keypoint, kept.keypoint) > overlap_threshold) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            result.push_back(candidate);
        }
    }
    return result;
}

cv::Mat loadHomography(const fs::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open homography: " + path.string());
    }
    cv::Mat H(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!(file >> H.at<double>(r, c))) {
                throw std::runtime_error("Invalid homography file: " + path.string());
            }
        }
    }
    return H;
}

std::vector<cv::Point2f> rotatedSquareCorners(const cv::Point2f& center,
                                              float side_length,
                                              float angle_rad) {
    const float half = side_length * 0.5f;
    std::vector<cv::Point2f> corners = {
        {-half, -half},
        {half, -half},
        {half, half},
        {-half, half}
    };
    const float cos_a = std::cos(angle_rad);
    const float sin_a = std::sin(angle_rad);
    for (auto& pt : corners) {
        const float x = pt.x;
        const float y = pt.y;
        pt.x = center.x + (x * cos_a - y * sin_a);
        pt.y = center.y + (x * sin_a + y * cos_a);
    }
    return corners;
}

float jitterValue(const std::map<std::string, std::vector<float>>& values,
                  const std::string& key,
                  int idx,
                  float default_value) {
    auto it = values.find(key);
    if (it == values.end()) {
        return default_value;
    }
    if (idx < 0 || idx >= static_cast<int>(it->second.size())) {
        return default_value;
    }
    return it->second[idx];
}

std::vector<cv::Point2f> jitteredSquareCorners(const cv::Point2f& center,
                                               float side_length,
                                               float base_angle_rad,
                                               float rot_jitter,
                                               float trans_jitter,
                                               float scale_jitter,
                                               float anis_jitter) {
    const float angle = base_angle_rad + rot_jitter;
    const float safe_anis = std::max(1e-6f, anis_jitter);
    const float scaled_side = side_length * scale_jitter;
    const float half_x = 0.5f * scaled_side * safe_anis;
    const float half_y = 0.5f * scaled_side / safe_anis;
    const float translate = trans_jitter * scaled_side * 0.1f;

    std::vector<cv::Point2f> corners = {
        {-half_x, -half_y},
        {half_x, -half_y},
        {half_x, half_y},
        {-half_x, half_y}
    };

    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);
    for (auto& pt : corners) {
        const float x = pt.x + translate;
        const float y = pt.y + translate;
        pt.x = center.x + (x * cos_a - y * sin_a);
        pt.y = center.y + (x * sin_a + y * cos_a);
    }
    return corners;
}

std::vector<cv::Point2f> projectCorners(const std::vector<cv::Point2f>& corners,
                                        const cv::Mat& H) {
    std::vector<cv::Point2f> projected;
    projected.reserve(corners.size());
    for (const auto& pt : corners) {
        const double x = pt.x;
        const double y = pt.y;
        const double w = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);
        const double px = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) / w;
        const double py = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) / w;
        projected.emplace_back(static_cast<float>(px), static_cast<float>(py));
    }
    return projected;
}

bool cornersInside(const std::vector<cv::Point2f>& corners, const cv::Size& size) {
    for (const auto& pt : corners) {
        if (pt.x < 0.0f || pt.y < 0.0f || pt.x >= size.width || pt.y >= size.height) {
            return false;
        }
    }
    return true;
}

cv::Mat extractPatch(const cv::Mat& image,
                     const std::vector<cv::Point2f>& corners) {
    const std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(kPatchSize - 1), 0.0f},
        {static_cast<float>(kPatchSize - 1), static_cast<float>(kPatchSize - 1)},
        {0.0f, static_cast<float>(kPatchSize - 1)}
    };
    const cv::Mat transform = cv::getPerspectiveTransform(corners, dst);
    cv::Mat patch;
    cv::warpPerspective(image, patch, transform, cv::Size(kPatchSize, kPatchSize),
                        cv::INTER_LINEAR, cv::BORDER_REFLECT101);
    return patch;
}

float polygonOverlapIoU(const std::vector<cv::Point2f>& a,
                         const std::vector<cv::Point2f>& b) {
    std::vector<cv::Point2f> intersection;
    const double inter_area = cv::intersectConvexConvex(a, b, intersection, true);
    const double area_a = std::fabs(cv::contourArea(a));
    const double area_b = std::fabs(cv::contourArea(b));
    const double union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(inter_area / union_area);
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    if (args.set_harris.empty() || args.set_hessian.empty() || args.set_dog.empty()) {
        std::cerr << "Error: --set-harris, --set-hessian, and --set-dog are required\n";
        printUsage(argv[0]);
        return 1;
    }

    thesis_project::database::DatabaseManager db(args.db_path, true);
    if (!db.isEnabled()) {
        std::cerr << "Failed to connect to database: " << args.db_path << "\n";
        return 1;
    }

    const int set_harris_id = db.getKeypointSetId(args.set_harris);
    const int set_hessian_id = db.getKeypointSetId(args.set_hessian);
    const int set_dog_id = db.getKeypointSetId(args.set_dog);
    if (set_harris_id < 0 || set_hessian_id < 0 || set_dog_id < 0) {
        std::cerr << "Failed to resolve keypoint set ids\n";
        return 1;
    }

    fs::create_directories(args.output_bw);
    fs::create_directories(args.output_color);

    const auto scenes = listScenes(args.images_dir);
    int scene_index = 0;
    for (const auto& scene : scenes) {
        scene_index++;
        if (args.verbose) {
            std::cout << "[" << scene_index << "/" << scenes.size() << "] " << scene << "\n";
        }

        const fs::path scene_dir = fs::path(args.images_dir) / scene;
        const fs::path metadata_scene = fs::path(args.metadata_dir) / scene;
        const fs::path out_bw_scene = fs::path(args.output_bw) / scene;
        const fs::path out_color_scene = fs::path(args.output_color) / scene;
        fs::create_directories(out_bw_scene);
        fs::create_directories(out_color_scene);

        SceneMetadata meta = loadSceneMetadata(metadata_scene, args.target_patches);
        const int target_total = meta.target_count;

        cv::Mat ref_color = cv::imread((scene_dir / "1.ppm").string(), cv::IMREAD_COLOR);
        if (ref_color.empty()) {
            std::cerr << "Failed to load reference image for " << scene << "\n";
            continue;
        }
        cv::Mat ref_gray;
        cv::cvtColor(ref_color, ref_gray, cv::COLOR_BGR2GRAY);

        auto harris_kps = db.getLockedKeypointsFromSet(set_harris_id, scene, "1.ppm");
        auto hessian_kps = db.getLockedKeypointsFromSet(set_hessian_id, scene, "1.ppm");
        auto dog_kps = db.getLockedKeypointsFromSet(set_dog_id, scene, "1.ppm");

        float weight_sum = args.weight_harris + args.weight_hessian + args.weight_dog;
        if (weight_sum <= 0.0f) {
            weight_sum = 1.0f;
        }
        const int harris_quota = static_cast<int>(std::ceil(target_total * (args.weight_harris / weight_sum)));
        const int hessian_quota = static_cast<int>(std::ceil(target_total * (args.weight_hessian / weight_sum)));
        const int dog_quota = static_cast<int>(std::ceil(target_total * (args.weight_dog / weight_sum)));

        const float mult = std::max(1.0f, args.candidate_multiplier);
        const int harris_limit = static_cast<int>(std::ceil(harris_quota * mult));
        const int hessian_limit = static_cast<int>(std::ceil(hessian_quota * mult));
        const int dog_limit = static_cast<int>(std::ceil(dog_quota * mult));

        if (static_cast<int>(harris_kps.size()) > harris_limit) harris_kps.resize(harris_limit);
        if (static_cast<int>(hessian_kps.size()) > hessian_limit) hessian_kps.resize(hessian_limit);
        if (static_cast<int>(dog_kps.size()) > dog_limit) dog_kps.resize(dog_limit);

        std::vector<DetectorKeypoint> merged;
        merged.reserve(harris_kps.size() + hessian_kps.size() + dog_kps.size());
        for (const auto& kp : harris_kps) merged.push_back({kp, "harris"});
        for (const auto& kp : hessian_kps) merged.push_back({kp, "hessian"});
        for (const auto& kp : dog_kps) merged.push_back({kp, "dog"});

        std::mt19937 rng(static_cast<unsigned>(1337 + scene_index));
        std::shuffle(merged.begin(), merged.end(), rng);

        // Load homographies (H_1_2 .. H_1_6) and target images (2..6)
        std::vector<cv::Mat> homographies;
        homographies.reserve(5);
        for (int i = 2; i <= 6; ++i) {
            homographies.push_back(loadHomography(scene_dir / ("H_1_" + std::to_string(i))));
        }

        std::vector<cv::Mat> target_images;
        target_images.reserve(5);
        for (int i = 2; i <= 6; ++i) {
            cv::Mat img = cv::imread((scene_dir / (std::to_string(i) + ".ppm")).string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Failed to load target image " << i << " for " << scene << "\n";
            }
            target_images.push_back(img);
        }

        std::vector<DetectorKeypoint> valid_candidates;
        std::vector<std::vector<cv::Point2f>> valid_ref_corners;
        valid_candidates.reserve(merged.size());
        valid_ref_corners.reserve(merged.size());

        for (int idx = 0; idx < static_cast<int>(merged.size()); ++idx) {
            const auto& candidate = merged[idx];
            const float angle_deg = candidate.keypoint.angle >= 0.0f ? candidate.keypoint.angle : 0.0f;
            const float angle_rad = angle_deg * static_cast<float>(CV_PI / 180.0);
            const float patch_side = candidate.keypoint.size * args.scale_multiplier;
            const auto ref_square = rotatedSquareCorners(candidate.keypoint.pt, patch_side, angle_rad);

            if (!cornersInside(ref_square, ref_color.size())) {
                continue;
            }

            bool valid = true;
            const std::vector<std::string> prefixes = {"e", "h", "t"};
            for (const auto& prefix : prefixes) {
                for (int img_idx = 0; img_idx < 5; ++img_idx) {
                    if (target_images[img_idx].empty()) {
                        valid = false;
                        break;
                    }
                    const std::string key = prefix + std::to_string(img_idx + 1);
                    const float rot_jitter = jitterValue(meta.rotjitter, key, idx, 0.0f);
                    const float trans_jitter = jitterValue(meta.trjitter, key, idx, 0.0f);
                    const float scale_jitter = jitterValue(meta.scjitter, key, idx, 1.0f);
                    const float anis_jitter = jitterValue(meta.anisjitter, key, idx, 1.0f);
                    const auto jittered_ref = jitteredSquareCorners(
                        candidate.keypoint.pt,
                        patch_side,
                        angle_rad,
                        rot_jitter,
                        trans_jitter,
                        scale_jitter,
                        anis_jitter);
                    const auto proj = projectCorners(jittered_ref, homographies[img_idx]);
                    if (!cornersInside(proj, target_images[img_idx].size())) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) {
                    break;
                }
            }

            if (!valid) {
                continue;
            }

            valid_candidates.push_back(candidate);
            valid_ref_corners.push_back(ref_square);
        }

        if (valid_candidates.empty()) {
            std::cerr << "No valid keypoints for scene " << scene << "\n";
            continue;
        }

        auto clustered = clusterOverlapping(valid_candidates, args.overlap_threshold);
        if (static_cast<int>(clustered.size()) > target_total) {
            clustered.resize(target_total);
        }

        std::vector<DetectorKeypoint> final_kps;
        std::vector<std::vector<cv::Point2f>> ref_corners;
        final_kps.reserve(clustered.size());
        ref_corners.reserve(clustered.size());

        for (const auto& candidate : clustered) {
            const float angle_deg = candidate.keypoint.angle >= 0.0f ? candidate.keypoint.angle : 0.0f;
            const float angle_rad = angle_deg * static_cast<float>(CV_PI / 180.0);
            const float patch_side = candidate.keypoint.size * args.scale_multiplier;
            ref_corners.push_back(rotatedSquareCorners(candidate.keypoint.pt, patch_side, angle_rad));
            final_kps.push_back(candidate);
        }

        // Prepare stacked outputs
        const int count = static_cast<int>(final_kps.size());
        cv::Mat ref_bw(kPatchSize * count, kPatchSize, CV_8UC1);
        cv::Mat ref_color_stack(kPatchSize * count, kPatchSize, CV_8UC3);

        auto writePatchRow = [&](const cv::Mat& patch, cv::Mat& stack, int idx) {
            patch.copyTo(stack(cv::Rect(0, idx * kPatchSize, kPatchSize, kPatchSize)));
        };

        for (int idx = 0; idx < count; ++idx) {
            const auto& corners = ref_corners[idx];
            cv::Mat patch_color = extractPatch(ref_color, corners);
            cv::Mat patch_bw;
            cv::cvtColor(patch_color, patch_bw, cv::COLOR_BGR2GRAY);
            writePatchRow(patch_bw, ref_bw, idx);
            writePatchRow(patch_color, ref_color_stack, idx);
        }

        cv::imwrite((out_bw_scene / "ref.png").string(), ref_bw);
        cv::imwrite((out_color_scene / "ref.png").string(), ref_color_stack);

        // Write ref metadata
        {
            std::ofstream overlaps(out_bw_scene / "ref.overlaps");
            std::ofstream overlaps_color(out_color_scene / "ref.overlaps");
            for (int i = 0; i < count; ++i) {
                overlaps << "1\n";
                overlaps_color << "1\n";
            }
            std::ofstream rotj(out_bw_scene / "ref.rotjitter");
            std::ofstream trj(out_bw_scene / "ref.trjitter");
            std::ofstream scj(out_bw_scene / "ref.scjitter");
            std::ofstream anisj(out_bw_scene / "ref.anisjitter");
            std::ofstream rotj_color(out_color_scene / "ref.rotjitter");
            std::ofstream trj_color(out_color_scene / "ref.trjitter");
            std::ofstream scj_color(out_color_scene / "ref.scjitter");
            std::ofstream anisj_color(out_color_scene / "ref.anisjitter");
            for (int i = 0; i < count; ++i) {
                rotj << "0\n";
                trj << "0\n";
                scj << "0\n";
                anisj << "0\n";
                rotj_color << "0\n";
                trj_color << "0\n";
                scj_color << "0\n";
                anisj_color << "0\n";
            }
        }

        // Copy homographies into output
        for (int i = 1; i <= 5; ++i) {
            const auto src = scene_dir / ("H_1_" + std::to_string(i + 1));
            const auto dst_bw = out_bw_scene / ("H_ref_" + std::to_string(i));
            const auto dst_color = out_color_scene / ("H_ref_" + std::to_string(i));
            if (fs::exists(src)) {
                fs::copy_file(src, dst_bw, fs::copy_options::overwrite_existing);
                fs::copy_file(src, dst_color, fs::copy_options::overwrite_existing);
            }
        }

        const std::vector<std::string> prefixes = {"e", "h", "t"};
        for (const auto& prefix : prefixes) {
            for (int img_idx = 0; img_idx < 5; ++img_idx) {
                const std::string key = prefix + std::to_string(img_idx + 1);
                const auto rotj = meta.rotjitter.find(key);
                const auto& jitter_vals = (rotj != meta.rotjitter.end()) ? rotj->second : std::vector<float>{};

                if (target_images[img_idx].empty()) {
                    continue;
                }

                cv::Mat stack_bw(kPatchSize * count, kPatchSize, CV_8UC1);
                cv::Mat stack_color(kPatchSize * count, kPatchSize, CV_8UC3);
                std::ofstream overlap_file_bw(out_bw_scene / (key + ".overlaps"));
                std::ofstream rotj_file_bw(out_bw_scene / (key + ".rotjitter"));
                std::ofstream trj_file_bw(out_bw_scene / (key + ".trjitter"));
                std::ofstream scj_file_bw(out_bw_scene / (key + ".scjitter"));
                std::ofstream anisj_file_bw(out_bw_scene / (key + ".anisjitter"));
                std::ofstream overlap_file_color(out_color_scene / (key + ".overlaps"));
                std::ofstream rotj_file_color(out_color_scene / (key + ".rotjitter"));
                std::ofstream trj_file_color(out_color_scene / (key + ".trjitter"));
                std::ofstream scj_file_color(out_color_scene / (key + ".scjitter"));
                std::ofstream anisj_file_color(out_color_scene / (key + ".anisjitter"));

                for (int idx = 0; idx < count; ++idx) {
                    const float rot_jitter = (idx < static_cast<int>(jitter_vals.size())) ? jitter_vals[idx] : 0.0f;
                    const float trans_jitter = jitterValue(meta.trjitter, key, idx, 0.0f);
                    const float scale_jitter = jitterValue(meta.scjitter, key, idx, 1.0f);
                    const float anis_jitter = jitterValue(meta.anisjitter, key, idx, 1.0f);
                    const float angle_deg = final_kps[idx].keypoint.angle >= 0.0f ? final_kps[idx].keypoint.angle : 0.0f;
                    const float angle_rad = angle_deg * static_cast<float>(CV_PI / 180.0f);
                    const float patch_side = final_kps[idx].keypoint.size * args.scale_multiplier;
                    const auto jittered_ref = jitteredSquareCorners(
                        final_kps[idx].keypoint.pt,
                        patch_side,
                        angle_rad,
                        rot_jitter,
                        trans_jitter,
                        scale_jitter,
                        anis_jitter);
                    const auto projected = projectCorners(jittered_ref, homographies[img_idx]);

                    cv::Mat patch_color = extractPatch(target_images[img_idx], projected);
                    cv::Mat patch_bw;
                    cv::cvtColor(patch_color, patch_bw, cv::COLOR_BGR2GRAY);
                    writePatchRow(patch_bw, stack_bw, idx);
                    writePatchRow(patch_color, stack_color, idx);

                    const float overlap = polygonOverlapIoU(
                        rotatedSquareCorners(final_kps[idx].keypoint.pt, patch_side,
                                             angle_deg * static_cast<float>(CV_PI / 180.0f)),
                        jittered_ref);
                    overlap_file_bw << overlap << "\n";
                    overlap_file_color << overlap << "\n";
                    rotj_file_bw << rot_jitter << "\n";
                    rotj_file_color << rot_jitter << "\n";
                    trj_file_bw << trans_jitter << "\n";
                    trj_file_color << trans_jitter << "\n";
                    scj_file_bw << scale_jitter << "\n";
                    scj_file_color << scale_jitter << "\n";
                    anisj_file_bw << anis_jitter << "\n";
                    anisj_file_color << anis_jitter << "\n";
                }

                cv::imwrite((out_bw_scene / (key + ".png")).string(), stack_bw);
                cv::imwrite((out_color_scene / (key + ".png")).string(), stack_color);
            }
        }
    }

    return 0;
}
