#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "cli/experiment_runner/processing.hpp"
#include "cli/experiment_runner/types.hpp"
#include "src/core/config/ExperimentConfig.hpp"

using thesis_project::config::ExperimentConfig;

namespace {

// Create a tiny HPatches-like scene with two images and an identity homography.
std::filesystem::path createTinyDataset() {
    namespace fs = std::filesystem;
    auto unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    auto tmp_root = fs::temp_directory_path() / ("dw_cli_processing_test_" + unique_suffix);
    fs::create_directories(tmp_root);
    auto scene_dir = tmp_root / "test_scene";
    fs::create_directories(scene_dir);

    cv::Mat img = cv::Mat::zeros(80, 80, CV_8UC3);
    cv::rectangle(img, cv::Point(10, 10), cv::Point(70, 70), cv::Scalar(200, 200, 200), cv::FILLED);
    cv::line(img, cv::Point(10, 40), cv::Point(70, 40), cv::Scalar(0, 0, 255), 2);
    cv::line(img, cv::Point(40, 10), cv::Point(40, 70), cv::Scalar(0, 255, 0), 2);
    std::vector<int> params = {cv::IMWRITE_PXM_BINARY, 1};
    cv::imwrite((scene_dir / "1.ppm").string(), img, params);
    cv::imwrite((scene_dir / "2.ppm").string(), img, params);

    std::ofstream hom_file(scene_dir / "H_1_2");
    // Identity homography
    hom_file << "1 0 0\n0 1 0\n0 0 1\n";
    hom_file.close();

    return tmp_root;
}

ExperimentConfig makeConfig(const std::filesystem::path& dataset_path) {
    ExperimentConfig cfg;
    cfg.experiment.name = "processing_smoke";
    cfg.dataset.path = dataset_path.string();
    cfg.keypoints.params.source = thesis_project::KeypointSource::HOMOGRAPHY_PROJECTION;
    cfg.evaluation.params.image_retrieval.enabled = false;
    cfg.evaluation.params.keypoint_verification.enabled = false;
    cfg.evaluation.params.keypoint_retrieval.enabled = false;
    cfg.performance.parallel_scenes = false;

    ExperimentConfig::DescriptorConfig desc;
    desc.name = "sift_smoke";
    desc.type = thesis_project::DescriptorType::SIFT;
    cfg.descriptors.push_back(desc);
    return cfg;
}

} // namespace

TEST(ExperimentRunnerProcessing, RunsOnTinyDataset) {
    auto dataset_root = createTinyDataset();
    ExperimentConfig cfg = makeConfig(dataset_root);

    ProfilingSummary profile;
    auto metrics = thesis_project::cli::experiment_runner_processing::processDirectoryNew(
        cfg,
        cfg.descriptors.front(),
        nullptr,
        -1,
        -1,
        profile);

    std::filesystem::remove_all(dataset_root);

    EXPECT_TRUE(metrics.success);
    EXPECT_GE(metrics.total_matches, 0);
    EXPECT_GE(profile.total_images, 1);
}
