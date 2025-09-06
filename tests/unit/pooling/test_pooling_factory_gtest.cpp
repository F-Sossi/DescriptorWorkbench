#include <gtest/gtest.h>
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/pooling/PoolingStrategy.hpp"
#include "src/core/pooling/NoPooling.hpp"
#include "src/core/pooling/DomainSizePooling.hpp"
#include "src/core/pooling/StackingPooling.hpp"
#include "src/core/config/experiment_config.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

using namespace thesis_project::pooling;

class PoolingFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a basic experiment config for testing
        config.descriptorOptions.poolingStrategy = NONE;
        config.descriptorOptions.descriptorColorSpace = D_BW;
        config.descriptorOptions.descriptorColorSpace2 = D_BW;
        
        // Create test image and keypoints
        test_image = cv::Mat::zeros(100, 100, CV_8UC3);  // Small color image
        test_keypoints = {
            cv::KeyPoint(25, 25, 10.0),
            cv::KeyPoint(75, 75, 15.0),
            cv::KeyPoint(50, 50, 12.0)
        };
        
        // Create primary detector (OpenCV SIFT)
        detector = cv::SIFT::create(100);  // Limited features for testing
        
        // Create secondary detector for stacking tests
        config.detector2 = cv::SIFT::create(50);  // Secondary detector
    }

    experiment_config config;
    cv::Mat test_image;
    std::vector<cv::KeyPoint> test_keypoints;
    cv::Ptr<cv::Feature2D> detector;
};

TEST_F(PoolingFactoryTest, CreateStrategyNone) {
    auto strategy = PoolingFactory::createStrategy(NONE);
    
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "None");
    EXPECT_EQ(strategy->getDimensionalityMultiplier(), 1.0f);
    EXPECT_FALSE(strategy->requiresColorInput());
}

TEST_F(PoolingFactoryTest, CreateStrategyDomainSize) {
    auto strategy = PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
    
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "DomainSizePooling");
    EXPECT_EQ(strategy->getDimensionalityMultiplier(), 1.0f);
    EXPECT_FALSE(strategy->requiresColorInput());
}

TEST_F(PoolingFactoryTest, CreateStrategyStacking) {
    auto strategy = PoolingFactory::createStrategy(STACKING);
    
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "Stacking");
    EXPECT_EQ(strategy->getDimensionalityMultiplier(), 2.0f);
    EXPECT_TRUE(strategy->requiresColorInput());  // Stacking typically requires color
}

TEST_F(PoolingFactoryTest, CreateStrategyInvalid) {
    // Test with invalid enum value (use global enum, not class name)
    EXPECT_THROW(
        PoolingFactory::createStrategy(static_cast<::PoolingStrategy>(999)),
        std::runtime_error
    );
}

TEST_F(PoolingFactoryTest, CreateFromConfig) {
    config.descriptorOptions.poolingStrategy = DOMAIN_SIZE_POOLING;
    
    auto strategy = PoolingFactory::createFromConfig(config);
    
    ASSERT_NE(strategy, nullptr);
    EXPECT_EQ(strategy->getName(), "DomainSizePooling");
}

TEST_F(PoolingFactoryTest, GetAvailableStrategies) {
    auto strategies = PoolingFactory::getAvailableStrategies();
    
    EXPECT_EQ(strategies.size(), 3);
    EXPECT_TRUE(std::find(strategies.begin(), strategies.end(), "None") != strategies.end());
    EXPECT_TRUE(std::find(strategies.begin(), strategies.end(), "DomainSizePooling") != strategies.end());
    EXPECT_TRUE(std::find(strategies.begin(), strategies.end(), "Stacking") != strategies.end());
}

TEST_F(PoolingFactoryTest, FactoryCreatesDifferentInstances) {
    auto strategy1 = PoolingFactory::createStrategy(NONE);
    auto strategy2 = PoolingFactory::createStrategy(NONE);
    
    ASSERT_NE(strategy1, nullptr);
    ASSERT_NE(strategy2, nullptr);
    EXPECT_NE(strategy1.get(), strategy2.get());  // Different instances
    EXPECT_EQ(strategy1->getName(), strategy2->getName());  // Same type
}

class PoolingStrategyIntegrationTest : public PoolingFactoryTest {
protected:
    void TearDown() override {
        // Clean up after OpenCV operations
    }
};

TEST_F(PoolingStrategyIntegrationTest, NoPoolingComputeDescriptors) {
    auto strategy = PoolingFactory::createStrategy(NONE);
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, test_keypoints.size());
    EXPECT_EQ(descriptors.cols, 128);  // SIFT descriptor size
    EXPECT_EQ(descriptors.type(), CV_32F);  // SIFT produces float descriptors
}

TEST_F(PoolingStrategyIntegrationTest, NoPoolingGrayscaleConversion) {
    auto strategy = PoolingFactory::createStrategy(NONE);
    
    // Test that color image gets converted to grayscale for BW descriptor
    config.descriptorOptions.descriptorColorSpace = D_BW;
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, test_keypoints.size());
}

TEST_F(PoolingStrategyIntegrationTest, DomainSizePoolingBasicComputation) {
    auto strategy = PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, test_keypoints.size());
    EXPECT_EQ(descriptors.cols, 128);  // DSP maintains same dimensionality
    EXPECT_EQ(descriptors.type(), CV_32F);
}

TEST_F(PoolingStrategyIntegrationTest, StackingPoolingBasicComputation) {
    auto strategy = PoolingFactory::createStrategy(STACKING);
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, test_keypoints.size());
    EXPECT_EQ(descriptors.cols, 256);  // Stacking doubles descriptor size (2x multiplier)
    EXPECT_EQ(descriptors.type(), CV_32F);
}

TEST_F(PoolingStrategyIntegrationTest, StrategyPolymorphism) {
    std::vector<::PoolingStrategy> strategies = {NONE, DOMAIN_SIZE_POOLING, STACKING};
    std::vector<std::string> expected_names = {"None", "DomainSizePooling", "Stacking"};
    std::vector<float> expected_multipliers = {1.0f, 1.0f, 2.0f};
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        auto strategy = PoolingFactory::createStrategy(strategies[i]);
        
        EXPECT_EQ(strategy->getName(), expected_names[i]);
        EXPECT_EQ(strategy->getDimensionalityMultiplier(), expected_multipliers[i]);
        
        // All strategies should be able to compute descriptors
        cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
        EXPECT_FALSE(descriptors.empty()) << "Strategy " << expected_names[i] << " failed";
        EXPECT_EQ(descriptors.rows, test_keypoints.size()) << "Strategy " << expected_names[i] << " row mismatch";
    }
}

TEST_F(PoolingStrategyIntegrationTest, EmptyKeypointsHandling) {
    auto strategy = PoolingFactory::createStrategy(NONE);
    std::vector<cv::KeyPoint> empty_keypoints;
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, empty_keypoints, detector, config);
    
    // Should return empty matrix for empty keypoints
    EXPECT_TRUE(descriptors.empty());
}

TEST_F(PoolingStrategyIntegrationTest, SingleKeypointHandling) {
    auto strategy = PoolingFactory::createStrategy(STACKING);
    std::vector<cv::KeyPoint> single_keypoint = {cv::KeyPoint(50, 50, 10.0)};
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, single_keypoint, detector, config);
    
    EXPECT_EQ(descriptors.rows, 1);
    EXPECT_EQ(descriptors.cols, 256);  // Stacking multiplier
}

// Test with different image types
TEST_F(PoolingStrategyIntegrationTest, GrayscaleImageHandling) {
    auto strategy = PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
    
    cv::Mat gray_image;
    cv::cvtColor(test_image, gray_image, cv::COLOR_BGR2GRAY);
    
    cv::Mat descriptors = strategy->computeDescriptors(gray_image, test_keypoints, detector, config);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, test_keypoints.size());
}

TEST_F(PoolingStrategyIntegrationTest, AllStrategiesBasicOperation) {
    std::vector<::PoolingStrategy> strategies = {NONE, DOMAIN_SIZE_POOLING, STACKING};
    std::vector<std::string> expected_names = {"None", "DomainSizePooling", "Stacking"};
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        auto strategy = PoolingFactory::createStrategy(strategies[i]);
        
        ASSERT_NE(strategy, nullptr) << "Failed to create strategy " << expected_names[i];
        
        // Basic properties should be consistent
        EXPECT_FALSE(strategy->getName().empty());
        EXPECT_GT(strategy->getDimensionalityMultiplier(), 0.0f);
        
        // Should be able to compute descriptors without throwing
        EXPECT_NO_THROW({
            cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
            if (!descriptors.empty()) {  // Some strategies might return empty for test conditions
                EXPECT_EQ(descriptors.rows, test_keypoints.size()) << "Strategy " << expected_names[i] << " row count mismatch";
            }
        });
    }
}

// Performance and edge case tests
TEST_F(PoolingStrategyIntegrationTest, LargeNumberOfKeypoints) {
    auto strategy = PoolingFactory::createStrategy(NONE);
    
    // Create many keypoints
    std::vector<cv::KeyPoint> many_keypoints;
    for (int i = 0; i < 50; ++i) {
        many_keypoints.emplace_back(i % 90 + 5, (i * 2) % 90 + 5, 10.0);
    }
    
    cv::Mat descriptors = strategy->computeDescriptors(test_image, many_keypoints, detector, config);
    
    EXPECT_LE(descriptors.rows, many_keypoints.size());  // Some keypoints may be rejected
    if (!descriptors.empty()) {
        EXPECT_EQ(descriptors.cols, 128);
    }
}

TEST_F(PoolingStrategyIntegrationTest, ConfigurationConsistency) {
    // Test that config changes are properly handled
    std::vector<DescriptorColorSpace> color_spaces = {D_BW};  // Only D_BW is defined
    
    for (auto color_space : color_spaces) {
        config.descriptorOptions.descriptorColorSpace = color_space;
        auto strategy = PoolingFactory::createStrategy(NONE);
        
        EXPECT_NO_THROW({
            cv::Mat descriptors = strategy->computeDescriptors(test_image, test_keypoints, detector, config);
            // Should handle different color spaces gracefully
        });
    }
}