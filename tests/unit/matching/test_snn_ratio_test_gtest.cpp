#include <gtest/gtest.h>
#include "src/core/matching/RatioTestMatching.hpp"
#include <opencv2/core.hpp>

using namespace thesis_project::matching;

// Test fixture for SNN Ratio Test Matching (tests only our custom code, not OpenCV)
class SNNRatioTestMatchingTest : public ::testing::Test {
protected:
    RatioTestMatching matcher{0.8f, cv::NORM_L2};
};

// ============================================================================
// Precision Calculation Tests (Our Custom Code)
// ============================================================================

TEST_F(SNNRatioTestMatchingTest, CalculatePrecision_EmptyMatches) {
    std::vector<cv::DMatch> emptyMatches;
    std::vector<cv::KeyPoint> keypoints = {cv::KeyPoint(100, 100, 10)};
    std::vector<cv::Point2f> projectedPoints = {cv::Point2f(100, 100)};

    double precision = matcher.calculatePrecision(emptyMatches, keypoints, projectedPoints, 3.0);

    EXPECT_DOUBLE_EQ(precision, 0.0) << "Empty matches should give 0 precision";
}

TEST_F(SNNRatioTestMatchingTest, CalculatePrecision_PerfectMatches) {
    // Create matches where projected points exactly match keypoints
    std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.5f),
        cv::DMatch(1, 1, 0.6f),
        cv::DMatch(2, 2, 0.4f)
    };

    std::vector<cv::KeyPoint> keypoints = {
        cv::KeyPoint(100, 100, 10),
        cv::KeyPoint(200, 200, 10),
        cv::KeyPoint(300, 300, 10)
    };

    std::vector<cv::Point2f> projectedPoints = {
        cv::Point2f(100, 100),  // Exact match
        cv::Point2f(200, 200),  // Exact match
        cv::Point2f(300, 300)   // Exact match
    };

    double precision = matcher.calculatePrecision(matches, keypoints, projectedPoints, 3.0);

    EXPECT_DOUBLE_EQ(precision, 1.0) << "Perfect matches should give precision = 1.0";
}

TEST_F(SNNRatioTestMatchingTest, CalculatePrecision_PartialCorrect) {
    std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.5f),  // Correct (distance = 0)
        cv::DMatch(1, 1, 0.6f),  // Incorrect (distance = 5)
        cv::DMatch(2, 2, 0.4f)   // Correct (distance = 2)
    };

    std::vector<cv::KeyPoint> keypoints = {
        cv::KeyPoint(100, 100, 10),
        cv::KeyPoint(200, 200, 10),
        cv::KeyPoint(300, 300, 10)
    };

    std::vector<cv::Point2f> projectedPoints = {
        cv::Point2f(100, 100),  // Distance = 0 (correct)
        cv::Point2f(205, 200),  // Distance = 5 (incorrect, > 3.0 threshold)
        cv::Point2f(302, 300)   // Distance = 2 (correct, < 3.0 threshold)
    };

    double precision = matcher.calculatePrecision(matches, keypoints, projectedPoints, 3.0);

    // 2 out of 3 matches are correct
    EXPECT_DOUBLE_EQ(precision, 2.0/3.0) << "Should calculate precision correctly for partial matches";
}

TEST_F(SNNRatioTestMatchingTest, CalculatePrecision_ThresholdSensitivity) {
    std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.5f)
    };

    std::vector<cv::KeyPoint> keypoints = {
        cv::KeyPoint(100, 100, 10)
    };

    std::vector<cv::Point2f> projectedPoints = {
        cv::Point2f(102.5, 102.5)  // Distance ~3.54
    };

    // With threshold 3.0, should be incorrect
    double precision1 = matcher.calculatePrecision(matches, keypoints, projectedPoints, 3.0);
    EXPECT_DOUBLE_EQ(precision1, 0.0) << "Distance > threshold should be incorrect";

    // With threshold 4.0, should be correct
    double precision2 = matcher.calculatePrecision(matches, keypoints, projectedPoints, 4.0);
    EXPECT_DOUBLE_EQ(precision2, 1.0) << "Distance < threshold should be correct";
}

TEST_F(SNNRatioTestMatchingTest, CalculatePrecision_InvalidIndices) {
    // Match with out-of-bounds indices
    std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.5f),
        cv::DMatch(10, 10, 0.6f),  // Invalid: index out of bounds
        cv::DMatch(1, 1, 0.4f)
    };

    std::vector<cv::KeyPoint> keypoints = {
        cv::KeyPoint(100, 100, 10),
        cv::KeyPoint(200, 200, 10)
    };

    std::vector<cv::Point2f> projectedPoints = {
        cv::Point2f(100, 100),
        cv::Point2f(200, 200)
    };

    double precision = matcher.calculatePrecision(matches, keypoints, projectedPoints, 3.0);

    // Invalid match should be skipped, so precision = 2 valid / 3 total = 0.666...
    EXPECT_DOUBLE_EQ(precision, 2.0/3.0) << "Invalid indices should be skipped in precision calculation";
}

// ============================================================================
// Match Threshold Adjustment Tests (Our Custom Code)
// ============================================================================

TEST_F(SNNRatioTestMatchingTest, AdjustMatchThreshold_ScaleFactor) {
    double baseThreshold = 3.0;

    // Scale factor 1.0 should return base threshold
    EXPECT_DOUBLE_EQ(matcher.adjustMatchThreshold(baseThreshold, 1.0), 3.0);

    // Scale factor 2.0 should double the threshold
    EXPECT_DOUBLE_EQ(matcher.adjustMatchThreshold(baseThreshold, 2.0), 6.0);

    // Scale factor 0.5 should halve the threshold
    EXPECT_DOUBLE_EQ(matcher.adjustMatchThreshold(baseThreshold, 0.5), 1.5);
}
