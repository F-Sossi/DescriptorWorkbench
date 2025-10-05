#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "include/thesis_project/types.hpp"

using thesis_project::pooling::PoolingFactory;
using thesis_project::config::ExperimentConfig;

namespace {
cv::Mat makeTestImage() {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(img, {50, 50}, 20, cv::Scalar(255, 255, 255), -1);
    return img;
}

std::vector<cv::KeyPoint> makeTestKeypoints() {
    return {
        cv::KeyPoint(25.f, 25.f, 10.f),
        cv::KeyPoint(75.f, 75.f, 12.f),
        cv::KeyPoint(50.f, 50.f, 14.f)
    };
}
}

class PoolingFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        descCfg.type = thesis_project::DescriptorType::SIFT;
        descCfg.name = "sift";
        descCfg.params.pooling = thesis_project::PoolingStrategy::NONE;
        descCfg.params.use_color = false;
        descCfg.params.normalize_after_pooling = true;
        extractor = thesis_project::factories::DescriptorFactory::create(descCfg.type);
        ASSERT_NE(extractor, nullptr);
        image = makeTestImage();
        keypoints = makeTestKeypoints();
    }

    ExperimentConfig::DescriptorConfig descCfg{};
    std::unique_ptr<thesis_project::IDescriptorExtractor> extractor;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
};

TEST_F(PoolingFactoryTest, CreateStrategyNone) {
    auto strategy = PoolingFactory::createStrategy(thesis_project::PoolingStrategy::NONE);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "None");
}

TEST_F(PoolingFactoryTest, CreateStrategyDomainSize) {
    auto strategy = PoolingFactory::createStrategy(thesis_project::PoolingStrategy::DOMAIN_SIZE_POOLING);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "DomainSizePooling");
}

TEST_F(PoolingFactoryTest, CreateStrategyStacking) {
    auto strategy = PoolingFactory::createStrategy(thesis_project::PoolingStrategy::STACKING);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "Stacking");
}

TEST_F(PoolingFactoryTest, CreateStrategyInvalidThrows) {
    auto invalid = static_cast<thesis_project::PoolingStrategy>(999);
    EXPECT_THROW(PoolingFactory::createStrategy(invalid), std::runtime_error);
}

TEST_F(PoolingFactoryTest, CreateFromDescriptorConfig) {
    descCfg.params.pooling = thesis_project::PoolingStrategy::DOMAIN_SIZE_POOLING;
    auto strategy = PoolingFactory::createFromConfig(descCfg);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "DomainSizePooling");
}

TEST_F(PoolingFactoryTest, GetAvailableStrategies) {
    auto strategies = PoolingFactory::getAvailableStrategies();
    EXPECT_EQ(strategies.size(), 3u);
    EXPECT_NE(std::find(strategies.begin(), strategies.end(), "None"), strategies.end());
}

TEST_F(PoolingFactoryTest, NoPoolingComputesDescriptors) {
    descCfg.params.pooling = thesis_project::PoolingStrategy::NONE;
    auto strategy = PoolingFactory::createFromConfig(descCfg);
    ASSERT_NE(strategy, nullptr);

    cv::Mat descriptors = strategy->computeDescriptors(image, keypoints, *extractor, descCfg);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, extractor->descriptorSize());
}

TEST_F(PoolingFactoryTest, DomainSizePoolingComputesDescriptors) {
    descCfg.params.pooling = thesis_project::PoolingStrategy::DOMAIN_SIZE_POOLING;
    descCfg.params.scales = {0.85f, 1.0f, 1.3f};
    auto strategy = PoolingFactory::createFromConfig(descCfg);
    ASSERT_NE(strategy, nullptr);

    cv::Mat descriptors = strategy->computeDescriptors(image, keypoints, *extractor, descCfg);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, extractor->descriptorSize());
}

TEST_F(PoolingFactoryTest, StackingPoolingConcatenatesDescriptors) {
    descCfg.params.pooling = thesis_project::PoolingStrategy::STACKING;
    descCfg.params.secondary_descriptor = thesis_project::DescriptorType::RGBSIFT;
    descCfg.params.use_color = false;

    auto strategy = PoolingFactory::createFromConfig(descCfg);
    ASSERT_NE(strategy, nullptr);

    cv::Mat descriptors = strategy->computeDescriptors(image, keypoints, *extractor, descCfg);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, extractor->descriptorSize() + 384); // RGBSIFT = 384 dims
}

