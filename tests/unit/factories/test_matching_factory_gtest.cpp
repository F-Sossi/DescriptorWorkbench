#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/matching/MatchingStrategy.hpp"
#include "include/thesis_project/types.hpp"

using thesis_project::matching::MatchingFactory;

class MatchingFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Minimal descriptor mats for a smoke test
        desc1 = cv::Mat::zeros(5, 8, CV_32F);
        desc2 = cv::Mat::zeros(5, 8, CV_32F);
        for (int r = 0; r < 5; ++r) {
            for (int c = 0; c < 8; ++c) {
                desc1.at<float>(r,c) = static_cast<float>(r + c);
                desc2.at<float>(r,c) = static_cast<float>(r + c);
            }
        }
    }

    cv::Mat desc1, desc2;
};

TEST_F(MatchingFactoryTest, CreateBruteForceStrategy) {
    auto strategy = MatchingFactory::createStrategy(thesis_project::MatchingMethod::BRUTE_FORCE);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "BruteForce");
}

TEST_F(MatchingFactoryTest, CreateRatioTestStrategy) {
    auto strategy = MatchingFactory::createStrategy(thesis_project::MatchingMethod::RATIO_TEST);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "RatioTest");

    auto matches = strategy->matchDescriptors(desc1, desc2);
    EXPECT_FALSE(matches.empty());
}

TEST_F(MatchingFactoryTest, CreateFLANNStrategy) {
    auto strategy = MatchingFactory::createStrategy(thesis_project::MatchingMethod::FLANN);
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "FLANN");
    EXPECT_TRUE(strategy->supportsRatioTest());

    auto matches = strategy->matchDescriptors(desc1, desc2);
    EXPECT_FALSE(matches.empty());
}

TEST_F(MatchingFactoryTest, CreateStrategyUnsupportedThrows) {
    EXPECT_NO_THROW(MatchingFactory::createStrategy(thesis_project::MatchingMethod::FLANN));
    auto unsupported = static_cast<thesis_project::MatchingMethod>(99);
    EXPECT_THROW(MatchingFactory::createStrategy(unsupported), std::runtime_error);
}

TEST_F(MatchingFactoryTest, GetAvailableStrategiesContainsBruteForce) {
    auto list = MatchingFactory::getAvailableStrategies();
    bool hasBF = false;
    for (const auto& name : list) if (name == "BruteForce") hasBF = true;
    EXPECT_TRUE(hasBF);
}
