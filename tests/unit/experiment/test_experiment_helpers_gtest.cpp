#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/experiment/ExperimentHelpers.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "src/core/matching/BruteForceMatching.hpp"
#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/pooling/PoolingStrategy.hpp"

using namespace thesis_project;

namespace {

class StubExtractor : public IDescriptorExtractor {
public:
    explicit StubExtractor(float value = 1.0f) : value_(value) {}

    cv::Mat extract(const cv::Mat& /*image*/, const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& /*params*/) override {
        return cv::Mat::ones(static_cast<int>(keypoints.size()), 4, CV_32F) * value_;
    }

    std::string name() const override { return "stub"; }
    int descriptorSize() const override { return 4; }
    int descriptorType() const override { return CV_32F; }

private:
    float value_;
};

class StubPooling : public pooling::PoolingStrategy {
public:
    cv::Mat computeDescriptors(const cv::Mat& image,
                               const std::vector<cv::KeyPoint>& keypoints,
                               IDescriptorExtractor& extractor,
                               const config::ExperimentConfig::DescriptorConfig& descCfg) override {
        lastImageSize = image.size();
        lastKeypointCount = keypoints.size();
        return extractor.extract(image, keypoints, descCfg.params);
    }

    std::string getName() const override { return "stub"; }
    float getDimensionalityMultiplier() const override { return 1.0f; }
    bool requiresColorInput() const override { return false; }

    cv::Size lastImageSize;
    size_t lastKeypointCount = 0;
};

std::vector<cv::KeyPoint> makeKeypoints(size_t count) {
    std::vector<cv::KeyPoint> kps;
    for (size_t i = 0; i < count; ++i) {
        kps.emplace_back(cv::Point2f(static_cast<float>(i), static_cast<float>(i)), 1.0f);
    }
    return kps;
}

} // namespace

TEST(ExperimentHelpersTest, ComputeDescriptorsWithPooling) {
    cv::Mat image = cv::Mat::ones(10, 10, CV_8UC1);
    auto keypoints = makeKeypoints(5);

    StubExtractor extractor(2.0f);
    StubPooling pooling;
    config::ExperimentConfig::DescriptorConfig descCfg;

    cv::Mat result = experiment::computeDescriptorsWithPooling(
        image, keypoints, extractor, pooling, descCfg);

    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(result.cols, extractor.descriptorSize());
    EXPECT_EQ(pooling.lastKeypointCount, keypoints.size());
    EXPECT_EQ(pooling.lastImageSize, image.size());
    EXPECT_NEAR(result.at<float>(0, 0), 2.0f, 1e-6);
}

TEST(ExperimentHelpersTest, ComputeMatchesEvaluatesCorrectness) {
    cv::Mat d1 = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);
    cv::Mat d2 = d1.clone();
    auto keypoints1 = makeKeypoints(2);
    auto keypoints2 = keypoints1;

    matching::BruteForceMatching matcher;
    auto artifacts = experiment::computeMatches(
        d1, d2, matcher, true, keypoints1, keypoints2);

    ASSERT_EQ(artifacts.matches.size(), 2u);
    EXPECT_EQ(artifacts.correctMatches, 2);
    EXPECT_TRUE(std::all_of(artifacts.correctnessFlags.begin(), artifacts.correctnessFlags.end(), [](bool f) { return f; }));

    auto artifactsNoEval = experiment::computeMatches(
        d1, d2, matcher, false, keypoints1, keypoints2);
    EXPECT_EQ(artifactsNoEval.correctMatches, 0);
    EXPECT_TRUE(std::all_of(artifactsNoEval.correctnessFlags.begin(), artifactsNoEval.correctnessFlags.end(), [](bool f) { return !f; }));
}

TEST(ExperimentHelpersTest, AccumulateTrueAveragePrecision) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    auto keypoints1 = makeKeypoints(2);
    auto keypoints2 = keypoints1;

    cv::Mat d1 = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
    cv::Mat d2 = d1.clone();

    ::ExperimentMetrics metrics;
    experiment::accumulateTrueAveragePrecision(H, keypoints1, d1, keypoints2, d2, "scene", metrics);
    metrics.calculateMeanPrecision();

    ASSERT_TRUE(metrics.per_scene_ap.count("scene"));
    EXPECT_EQ(metrics.per_scene_ap.at("scene").size(), keypoints1.size());
}

