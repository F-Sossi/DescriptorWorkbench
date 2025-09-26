#include <gtest/gtest.h>
#include "src/core/keypoints/detectors/SIFTKeypointGenerator.hpp"
#include "src/core/keypoints/detectors/HarrisKeypointGenerator.hpp"
#include "src/core/keypoints/detectors/ORBKeypointGenerator.hpp"
#include "src/core/keypoints/detectors/NonOverlappingKeypointGenerator.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <opencv2/imgproc.hpp>

using namespace thesis_project;

class KeypointGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple test image (100x100 with some features)
        test_image_ = cv::Mat::zeros(100, 100, CV_8UC1);
        
        // Add some corners/features for detection
        cv::rectangle(test_image_, cv::Rect(20, 20, 30, 30), cv::Scalar(255), 2);
        cv::rectangle(test_image_, cv::Rect(60, 60, 25, 25), cv::Scalar(255), 2);
        cv::circle(test_image_, cv::Point(30, 70), 10, cv::Scalar(255), 1);
        
        // Set up parameters
        params_.max_features = 100;
        params_.contrast_threshold = 0.04f;
        params_.edge_threshold = 10.0f;
    }
    
    cv::Mat test_image_;
    KeypointParams params_;
    static constexpr float border_ = 35.0f;
};

TEST_F(KeypointGeneratorTest, SIFTDetectorBasicFunctionality) {
    SIFTKeypointGenerator sift_detector;
    
    // Test basic properties
    EXPECT_EQ(sift_detector.name(), "SIFT");
    EXPECT_EQ(sift_detector.type(), KeypointGenerator::SIFT);
    EXPECT_TRUE(sift_detector.supportsNonOverlapping());
    
    // Test detection
    auto keypoints = sift_detector.detect(test_image_, params_);
    EXPECT_FALSE(keypoints.empty()) << "SIFT should detect some keypoints in test image";
    
    // Test all keypoints are within bounds (35px border, matching generator filter)
    for (const auto& kp : keypoints) {
        EXPECT_GE(kp.pt.x, border_);
        EXPECT_GE(kp.pt.y, border_);
        EXPECT_LT(kp.pt.x, test_image_.cols - border_);
        EXPECT_LT(kp.pt.y, test_image_.rows - border_);
    }
}

TEST_F(KeypointGeneratorTest, HarrisDetectorBasicFunctionality) {
    HarrisKeypointGenerator harris_detector;
    
    // Test basic properties
    EXPECT_EQ(harris_detector.name(), "Harris");
    EXPECT_EQ(harris_detector.type(), KeypointGenerator::HARRIS);
    EXPECT_TRUE(harris_detector.supportsNonOverlapping());
    
    // Test detection
    auto keypoints = harris_detector.detect(test_image_, params_);
    EXPECT_FALSE(keypoints.empty()) << "Harris should detect some keypoints in test image";
    
    // Test boundary filtering
    for (const auto& kp : keypoints) {
        EXPECT_GE(kp.pt.x, border_);
        EXPECT_GE(kp.pt.y, border_);
        EXPECT_LT(kp.pt.x, test_image_.cols - border_);
        EXPECT_LT(kp.pt.y, test_image_.rows - border_);
    }
}

TEST_F(KeypointGeneratorTest, ORBDetectorBasicFunctionality) {
    ORBKeypointGenerator orb_detector;
    
    // Test basic properties
    EXPECT_EQ(orb_detector.name(), "ORB");
    EXPECT_EQ(orb_detector.type(), KeypointGenerator::ORB);
    EXPECT_TRUE(orb_detector.supportsNonOverlapping());
    
    // Test detection
    auto keypoints = orb_detector.detect(test_image_, params_);
    EXPECT_FALSE(keypoints.empty()) << "ORB should detect some keypoints in test image";
    
    // Test boundary filtering
    for (const auto& kp : keypoints) {
        EXPECT_GE(kp.pt.x, border_);
        EXPECT_GE(kp.pt.y, border_);
        EXPECT_LT(kp.pt.x, test_image_.cols - border_);
        EXPECT_LT(kp.pt.y, test_image_.rows - border_);
    }
}

TEST_F(KeypointGeneratorTest, NonOverlappingDecorator) {
    auto base_detector = std::make_unique<SIFTKeypointGenerator>();
    float min_distance = 20.0f;
    
    NonOverlappingKeypointGenerator non_overlapping_detector(
        std::move(base_detector), min_distance);
    
    // Test properties
    EXPECT_EQ(non_overlapping_detector.name(), "SIFT_NonOverlapping");
    EXPECT_EQ(non_overlapping_detector.type(), KeypointGenerator::SIFT);
    EXPECT_TRUE(non_overlapping_detector.supportsNonOverlapping());
    EXPECT_EQ(non_overlapping_detector.getDefaultMinDistance(), min_distance);
    
    // Test non-overlapping detection
    auto keypoints = non_overlapping_detector.detectNonOverlapping(test_image_, min_distance, params_);
    
    // Verify no keypoints are closer than min_distance
    for (size_t i = 0; i < keypoints.size(); ++i) {
        for (size_t j = i + 1; j < keypoints.size(); ++j) {
            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            
            EXPECT_GE(distance, min_distance) 
                << "Keypoints " << i << " and " << j << " are too close: " << distance;
        }
    }
}

TEST_F(KeypointGeneratorTest, FactoryPatternSIFT) {
    auto detector = KeypointGeneratorFactory::create(KeypointGenerator::SIFT);
    
    EXPECT_NE(detector, nullptr);
    EXPECT_EQ(detector->name(), "SIFT");
    EXPECT_EQ(detector->type(), KeypointGenerator::SIFT);
    
    auto keypoints = detector->detect(test_image_, params_);
    EXPECT_FALSE(keypoints.empty());
}

TEST_F(KeypointGeneratorTest, FactoryPatternWithNonOverlapping) {
    float min_distance = 15.0f;
    auto detector = KeypointGeneratorFactory::create(
        KeypointGenerator::SIFT, true, min_distance);
    
    EXPECT_NE(detector, nullptr);
    EXPECT_EQ(detector->name(), "SIFT_NonOverlapping");
    EXPECT_TRUE(detector->supportsNonOverlapping());
    
    auto keypoints = detector->detectNonOverlapping(test_image_, min_distance, params_);
    
    // Verify non-overlapping constraint
    for (size_t i = 0; i < keypoints.size(); ++i) {
        for (size_t j = i + 1; j < keypoints.size(); ++j) {
            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            EXPECT_GE(distance, min_distance);
        }
    }
}

TEST_F(KeypointGeneratorTest, FactoryStringParsing) {
    // Test string parsing
    EXPECT_EQ(KeypointGeneratorFactory::parseDetectorType("sift"), KeypointGenerator::SIFT);
    EXPECT_EQ(KeypointGeneratorFactory::parseDetectorType("SIFT"), KeypointGenerator::SIFT);
    EXPECT_EQ(KeypointGeneratorFactory::parseDetectorType("harris"), KeypointGenerator::HARRIS);
    EXPECT_EQ(KeypointGeneratorFactory::parseDetectorType("orb"), KeypointGenerator::ORB);
    
    // Test invalid string
    EXPECT_THROW(KeypointGeneratorFactory::parseDetectorType("invalid"), std::invalid_argument);
}

TEST_F(KeypointGeneratorTest, FactoryCreateFromString) {
    auto detector = KeypointGeneratorFactory::createFromString("harris");
    
    EXPECT_NE(detector, nullptr);
    EXPECT_EQ(detector->name(), "Harris");
    EXPECT_EQ(detector->type(), KeypointGenerator::HARRIS);
}

TEST_F(KeypointGeneratorTest, FactorySupportedDetectors) {
    auto supported = KeypointGeneratorFactory::getSupportedDetectors();
    
    EXPECT_FALSE(supported.empty());
    EXPECT_TRUE(std::find(supported.begin(), supported.end(), "sift") != supported.end());
    EXPECT_TRUE(std::find(supported.begin(), supported.end(), "harris") != supported.end());
    EXPECT_TRUE(std::find(supported.begin(), supported.end(), "orb") != supported.end());
}

TEST_F(KeypointGeneratorTest, RecommendedMinDistances) {
    // Test recommended distances for different detectors
    float sift_dist = KeypointGeneratorFactory::getRecommendedMinDistance(KeypointGenerator::SIFT, 32);
    float harris_dist = KeypointGeneratorFactory::getRecommendedMinDistance(KeypointGenerator::HARRIS, 32);
    float orb_dist = KeypointGeneratorFactory::getRecommendedMinDistance(KeypointGenerator::ORB, 32);
    
    EXPECT_EQ(sift_dist, 32.0f);
    EXPECT_EQ(harris_dist, 32.0f * 0.8f);  // Harris uses 0.8 factor
    EXPECT_EQ(orb_dist, 32.0f);  // ORB uses max(32, 31) = 32
}

TEST_F(KeypointGeneratorTest, EmptyImageHandling) {
    cv::Mat empty_image;
    SIFTKeypointGenerator detector;
    
    auto keypoints = detector.detect(empty_image, params_);
    EXPECT_TRUE(keypoints.empty()) << "Empty image should produce no keypoints";
}

TEST_F(KeypointGeneratorTest, FeatureLimiting) {
    SIFTKeypointGenerator detector;
    
    // Set very low feature limit
    KeypointParams limited_params = params_;
    limited_params.max_features = 2;
    
    auto keypoints = detector.detect(test_image_, limited_params);
    EXPECT_LE(keypoints.size(), 2u) << "Should respect max_features limit";
}

TEST_F(KeypointGeneratorTest, NonOverlappingReducesKeypoints) {
    auto base_detector = std::make_unique<SIFTKeypointGenerator>();
    
    // Get baseline keypoints
    auto baseline_keypoints = base_detector->detect(test_image_, params_);
    
    // Create non-overlapping version
    auto non_overlapping = KeypointGeneratorFactory::makeNonOverlapping(
        std::make_unique<SIFTKeypointGenerator>(), 25.0f);
    
    auto filtered_keypoints = non_overlapping->detectNonOverlapping(test_image_, 25.0f, params_);
    
    // Non-overlapping should typically produce fewer keypoints
    EXPECT_LE(filtered_keypoints.size(), baseline_keypoints.size())
        << "Non-overlapping filtering should reduce or maintain keypoint count";
}

// Integration test to ensure all components work together
TEST_F(KeypointGeneratorTest, EndToEndWorkflow) {
    // Create detector using factory
    auto detector = KeypointGeneratorFactory::createFromString("sift", true, 20.0f);
    
    // Detect keypoints
    auto keypoints = detector->detect(test_image_, params_);
    
    // Verify we got valid results
    EXPECT_FALSE(keypoints.empty());
    
    // Verify properties
    for (const auto& kp : keypoints) {
        EXPECT_GT(kp.response, 0.0f) << "Keypoint should have positive response";
        EXPECT_GE(kp.size, 0.0f) << "Keypoint size should be non-negative";
    }
    
    // Verify non-overlapping constraint was applied
    float min_distance = 20.0f;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        for (size_t j = i + 1; j < keypoints.size(); ++j) {
            float dx = keypoints[i].pt.x - keypoints[j].pt.x;
            float dy = keypoints[i].pt.y - keypoints[j].pt.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            EXPECT_GE(distance, min_distance);
        }
    }
}
