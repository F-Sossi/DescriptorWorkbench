#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <vector>

#include "src/core/descriptor/extractors/CompositeDescriptorExtractor.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include "thesis_project/types.hpp"

namespace {

std::filesystem::path makeTempDbPath() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto tmp_dir = std::filesystem::temp_directory_path();
    return tmp_dir / ("composite_tls_" + std::to_string(now) + ".db");
}

std::vector<cv::KeyPoint> makeKeypoints(float offset, int count) {
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(count);
    for (int i = 0; i < count; ++i) {
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(offset + static_cast<float>(i * 3), offset + static_cast<float>(i * 2));
        kp.size = 5.0f;
        kp.angle = 0.0f;
        kp.response = 1.0f;
        kp.octave = 0;
        kp.class_id = i;
        keypoints.push_back(kp);
    }
    return keypoints;
}

struct TempDatabaseFile {
    std::filesystem::path path;
    TempDatabaseFile() : path(makeTempDbPath()) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    ~TempDatabaseFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

} // namespace

TEST(CompositeThreadSafety, ThreadLocalDatabaseContextPreventsCrossTalk) {
    TempDatabaseFile temp_db;
    thesis_project::database::DatabaseManager db(temp_db.path.string(), true);
    ASSERT_TRUE(db.isEnabled());

    const int set_a = db.createKeypointSet("tls_set_a", "sift", "independent_detection");
    const int set_b = db.createKeypointSet("tls_set_b", "sift", "independent_detection");
    ASSERT_GT(set_a, 0);
    ASSERT_GT(set_b, 0);

    const std::string image_name = "1.ppm";
    struct SceneInfo {
        std::string name;
        int count;
    };
    std::vector<SceneInfo> scenes = {
        {"scene_parallel_a", 3},
        {"scene_parallel_b", 5},
    };

    for (const auto& scene : scenes) {
        auto kps_a = makeKeypoints(5.0f, scene.count);
        auto kps_b = makeKeypoints(15.0f, scene.count);
        ASSERT_TRUE(db.storeLockedKeypointsForSet(set_a, scene.name, image_name, kps_a));
        ASSERT_TRUE(db.storeLockedKeypointsForSet(set_b, scene.name, image_name, kps_b));
    }

    using CompositeConfig = thesis_project::CompositeDescriptorExtractor::ComponentConfig;
    std::vector<CompositeConfig> components;
    CompositeConfig comp_a(thesis_project::DescriptorType::SIFT, 0.5);
    comp_a.keypoint_set_name = "tls_set_a";
    CompositeConfig comp_b(thesis_project::DescriptorType::SIFT, 0.5);
    comp_b.keypoint_set_name = "tls_set_b";
    components.push_back(comp_a);
    components.push_back(comp_b);

    thesis_project::CompositeDescriptorExtractor extractor(
        std::move(components),
        thesis_project::CompositeDescriptorExtractor::AggregationMethod::WEIGHTED_AVG);
    ASSERT_TRUE(extractor.usesPairedKeypointSets());

    thesis_project::DescriptorParams params;
    params.pooling = thesis_project::PoolingStrategy::NONE;

    cv::Mat image = cv::Mat::zeros(64, 64, CV_8UC1);

    std::unordered_map<std::string, std::vector<cv::KeyPoint>> fallback_keypoints;
    for (const auto& scene : scenes) {
        auto shared = db.getLockedKeypointsFromSet(set_a, scene.name, image_name);
        ASSERT_FALSE(shared.empty());
        fallback_keypoints.emplace(scene.name, shared);
    }

    auto worker = [&](const SceneInfo& scene) {
        for (int iter = 0; iter < 3; ++iter) {
            extractor.setDatabaseContext(&db, scene.name, image_name);
            const auto& shared_keypoints = fallback_keypoints.at(scene.name);
            cv::Mat descriptors = extractor.extract(image, shared_keypoints, params);
            EXPECT_EQ(descriptors.rows, scene.count);
            EXPECT_EQ(descriptors.cols, extractor.descriptorSize());
            EXPECT_EQ(descriptors.type(), CV_32F);
        }
    };

    std::thread t1(worker, scenes[0]);
    std::thread t2(worker, scenes[1]);
    t1.join();
    t2.join();
}
