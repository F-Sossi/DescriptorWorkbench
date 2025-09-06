#include <gtest/gtest.h>
#include "src/core/metrics/MetricsCalculator.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include <vector>
#include <chrono>
#include <thread>

class MetricsCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create sample folder metrics for aggregation testing
        folder_metrics1.success = true;
        folder_metrics1.addImageResult("scene1", 0.8, 40, 80);
        folder_metrics1.addImageResult("scene1", 0.6, 30, 60);
        folder_metrics1.addImageResult("scene2", 0.9, 30, 60);

        folder_metrics2.success = true;
        folder_metrics2.addImageResult("scene2", 0.7, 25, 50);
        folder_metrics2.addImageResult("scene3", 0.5, 25, 50);

        // Failed metrics for error handling
        failed_metrics.success = false;
        failed_metrics.error_message = "Processing failed";
    }

    ExperimentMetrics folder_metrics1;
    ExperimentMetrics folder_metrics2;
    ExperimentMetrics failed_metrics;
};

TEST_F(MetricsCalculatorTest, AggregateMetricsBasic) {
    std::vector<ExperimentMetrics> folder_metrics = {folder_metrics1, folder_metrics2};
    double processing_time = 1500.0;
    
    ExperimentMetrics result = MetricsCalculator::aggregateMetrics(folder_metrics, processing_time);
    
    // Check basic aggregation
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.processing_time_ms, processing_time);
    EXPECT_EQ(result.total_images_processed, 5);  // 3 + 2
    EXPECT_EQ(result.total_matches, 150);         // 100 + 50
    EXPECT_EQ(result.total_keypoints, 300);       // 200 + 100
    
    // Check precision vector merging
    ASSERT_EQ(result.precisions_per_image.size(), 5);
    std::vector<double> expected_precisions = {0.8, 0.6, 0.9, 0.7, 0.5};
    for (size_t i = 0; i < expected_precisions.size(); ++i) {
        EXPECT_DOUBLE_EQ(result.precisions_per_image[i], expected_precisions[i]);
    }
}

TEST_F(MetricsCalculatorTest, AggregateMetricsWithFailures) {
    std::vector<ExperimentMetrics> folder_metrics = {folder_metrics1, failed_metrics, folder_metrics2};
    double processing_time = 2000.0;
    
    ExperimentMetrics result = MetricsCalculator::aggregateMetrics(folder_metrics, processing_time);
    
    // Should be marked as failed due to one failure
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.error_message, "Processing failed");
    EXPECT_EQ(result.processing_time_ms, processing_time);
    
    // Should still aggregate the successful metrics
    EXPECT_EQ(result.total_images_processed, 5);  // Only successful ones: 3 + 2
    EXPECT_EQ(result.total_matches, 150);
    EXPECT_EQ(result.total_keypoints, 300);
}

TEST_F(MetricsCalculatorTest, AggregateMetricsEmpty) {
    std::vector<ExperimentMetrics> empty_metrics;
    double processing_time = 500.0;
    
    ExperimentMetrics result = MetricsCalculator::aggregateMetrics(empty_metrics, processing_time);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.processing_time_ms, processing_time);
    EXPECT_EQ(result.total_images_processed, 0);
    EXPECT_EQ(result.total_matches, 0);
    EXPECT_EQ(result.total_keypoints, 0);
    EXPECT_TRUE(result.precisions_per_image.empty());
}

TEST_F(MetricsCalculatorTest, AggregateMetricsPerSceneMerging) {
    std::vector<ExperimentMetrics> folder_metrics = {folder_metrics1, folder_metrics2};
    
    ExperimentMetrics result = MetricsCalculator::aggregateMetrics(folder_metrics, 1000.0);
    
    // Check per-scene merging - scene1 should have 2 results, scene2 should have 2, scene3 should have 1
    EXPECT_EQ(result.per_scene_precisions.size(), 3);
    
    // scene1: from folder_metrics1 only (2 results)
    ASSERT_EQ(result.per_scene_precisions["scene1"].size(), 2);
    EXPECT_DOUBLE_EQ(result.per_scene_precisions["scene1"][0], 0.8);
    EXPECT_DOUBLE_EQ(result.per_scene_precisions["scene1"][1], 0.6);
    
    // scene2: from both folder metrics (2 results total)
    ASSERT_EQ(result.per_scene_precisions["scene2"].size(), 2);
    EXPECT_DOUBLE_EQ(result.per_scene_precisions["scene2"][0], 0.9);  // From folder_metrics1
    EXPECT_DOUBLE_EQ(result.per_scene_precisions["scene2"][1], 0.7);  // From folder_metrics2
    
    // scene3: from folder_metrics2 only (1 result)
    ASSERT_EQ(result.per_scene_precisions["scene3"].size(), 1);
    EXPECT_DOUBLE_EQ(result.per_scene_precisions["scene3"][0], 0.5);
}

TEST_F(MetricsCalculatorTest, CalculateProcessingTime) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate some processing time (small delay)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = MetricsCalculator::calculateProcessingTime(start, end);
    
    // Should be at least 10ms, but allow some variance due to system timing
    EXPECT_GE(elapsed, 8.0);   // At least 8ms (allowing for timing variance)
    EXPECT_LE(elapsed, 50.0);  // But not more than 50ms (reasonable upper bound)
}

TEST_F(MetricsCalculatorTest, CalculateProcessingTimeZero) {
    auto time_point = std::chrono::high_resolution_clock::now();
    
    double elapsed = MetricsCalculator::calculateProcessingTime(time_point, time_point);
    
    EXPECT_EQ(elapsed, 0.0);
}

TEST_F(MetricsCalculatorTest, CalculatePrecisionBasic) {
    // Perfect precision
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(100, 100), 1.0);
    
    // 50% precision
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(100, 50), 0.5);
    
    // No correct matches
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(100, 0), 0.0);
    
    // Typical case
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(75, 45), 0.6);
}

TEST_F(MetricsCalculatorTest, CalculatePrecisionEdgeCases) {
    // No total matches - should return 0.0
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(0, 5), 0.0);  // Invalid case but shouldn't crash
    
    // Single match cases
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecision(1, 0), 0.0);
}

TEST_F(MetricsCalculatorTest, CalculatePrecisionFromMatchesBasic) {
    // All correct matches
    std::vector<bool> all_correct = {true, true, true, true};
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(all_correct, 4), 1.0);
    
    // Half correct matches  
    std::vector<bool> half_correct = {true, false, true, false};
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(half_correct, 2), 0.5);
    
    // No correct matches
    std::vector<bool> no_correct = {false, false, false};
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(no_correct, 0), 0.0);
    
    // Single match
    std::vector<bool> single_match = {true};
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(single_match, 1), 1.0);
}

TEST_F(MetricsCalculatorTest, CalculatePrecisionFromMatchesEdgeCases) {
    // Empty vector
    std::vector<bool> empty_matches;
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(empty_matches, 0), 0.0);
    
    // Inconsistent count (should use vector size, not count parameter for denominator)
    std::vector<bool> matches = {true, false, true};  // 3 total, but count says 2
    EXPECT_DOUBLE_EQ(MetricsCalculator::calculatePrecisionFromMatches(matches, 2), 2.0/3.0);
}

// Parameterized test for precision calculations
class PrecisionParameterizedTest : public ::testing::TestWithParam<std::tuple<int, int, double>> {};

TEST_P(PrecisionParameterizedTest, CalculatePrecisionVariousCases) {
    int total = std::get<0>(GetParam());
    int correct = std::get<1>(GetParam());
    double expected = std::get<2>(GetParam());
    
    double result = MetricsCalculator::calculatePrecision(total, correct);
    EXPECT_DOUBLE_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionTests,
    PrecisionParameterizedTest,
    ::testing::Values(
        std::make_tuple(100, 75, 0.75),
        std::make_tuple(50, 25, 0.5),
        std::make_tuple(200, 160, 0.8),
        std::make_tuple(10, 3, 0.3),
        std::make_tuple(1, 1, 1.0),
        std::make_tuple(1000, 0, 0.0)
    )
);