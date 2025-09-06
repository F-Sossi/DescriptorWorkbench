#include <gtest/gtest.h>
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "src/core/metrics/TrueAveragePrecision.hpp"
#include <vector>
#include <map>
#include <string>

class ExperimentMetricsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create clean metrics object for each test
        metrics = ExperimentMetrics();
        metrics.success = true;
    }

    ExperimentMetrics metrics;
};

TEST_F(ExperimentMetricsTest, DefaultConstruction) {
    ExperimentMetrics default_metrics;
    
    EXPECT_TRUE(default_metrics.success);
    EXPECT_EQ(default_metrics.mean_precision, 0.0);
    EXPECT_EQ(default_metrics.legacy_macro_precision_by_scene, 0.0);
    EXPECT_EQ(default_metrics.total_matches, 0);
    EXPECT_EQ(default_metrics.total_keypoints, 0);
    EXPECT_EQ(default_metrics.total_images_processed, 0);
    EXPECT_EQ(default_metrics.processing_time_ms, 0.0);
    EXPECT_TRUE(default_metrics.precisions_per_image.empty());
    EXPECT_TRUE(default_metrics.per_scene_precisions.empty());
    EXPECT_TRUE(default_metrics.error_message.empty());
}

TEST_F(ExperimentMetricsTest, AddImageResultBasic) {
    metrics.addImageResult("test_scene", 0.75, 30, 40);
    
    EXPECT_EQ(metrics.total_images_processed, 1);
    EXPECT_EQ(metrics.total_matches, 30);
    EXPECT_EQ(metrics.total_keypoints, 40);
    ASSERT_EQ(metrics.precisions_per_image.size(), 1);
    EXPECT_DOUBLE_EQ(metrics.precisions_per_image[0], 0.75);
    
    // Check per-scene data
    ASSERT_EQ(metrics.per_scene_precisions.count("test_scene"), 1);
    ASSERT_EQ(metrics.per_scene_precisions["test_scene"].size(), 1);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["test_scene"][0], 0.75);
    EXPECT_EQ(metrics.per_scene_matches["test_scene"], 30);
    EXPECT_EQ(metrics.per_scene_keypoints["test_scene"], 40);
    EXPECT_EQ(metrics.per_scene_image_count["test_scene"], 1);
}

TEST_F(ExperimentMetricsTest, AddImageResultMultipleImages) {
    metrics.addImageResult("scene1", 0.8, 25, 50);
    metrics.addImageResult("scene1", 0.6, 15, 30);
    metrics.addImageResult("scene2", 0.9, 35, 70);
    
    EXPECT_EQ(metrics.total_images_processed, 3);
    EXPECT_EQ(metrics.total_matches, 75);  // 25 + 15 + 35
    EXPECT_EQ(metrics.total_keypoints, 150); // 50 + 30 + 70
    
    // Check scene1
    ASSERT_EQ(metrics.per_scene_precisions["scene1"].size(), 2);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["scene1"][0], 0.8);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["scene1"][1], 0.6);
    EXPECT_EQ(metrics.per_scene_matches["scene1"], 40);  // 25 + 15
    EXPECT_EQ(metrics.per_scene_keypoints["scene1"], 80); // 50 + 30
    EXPECT_EQ(metrics.per_scene_image_count["scene1"], 2);
    
    // Check scene2
    ASSERT_EQ(metrics.per_scene_precisions["scene2"].size(), 1);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["scene2"][0], 0.9);
    EXPECT_EQ(metrics.per_scene_matches["scene2"], 35);
    EXPECT_EQ(metrics.per_scene_keypoints["scene2"], 70);
    EXPECT_EQ(metrics.per_scene_image_count["scene2"], 1);
}

TEST_F(ExperimentMetricsTest, AddQueryAPWithMatch) {
    TrueAveragePrecision::QueryAPResult ap_result;
    ap_result.ap = 0.8;
    ap_result.has_potential_match = true;
    ap_result.rank_of_true_match = 2;
    ap_result.total_relevant = 1;
    
    metrics.addQueryAP("test_scene", ap_result);
    
    EXPECT_EQ(metrics.total_queries_processed, 1);
    EXPECT_EQ(metrics.total_queries_excluded, 0);
    ASSERT_EQ(metrics.ap_per_query.size(), 1);
    EXPECT_DOUBLE_EQ(metrics.ap_per_query[0], 0.8);
    ASSERT_EQ(metrics.per_scene_ap["test_scene"].size(), 1);
    EXPECT_DOUBLE_EQ(metrics.per_scene_ap["test_scene"][0], 0.8);
    ASSERT_EQ(metrics.ranks_per_query.size(), 1);
    EXPECT_EQ(metrics.ranks_per_query[0], 2);
}

TEST_F(ExperimentMetricsTest, AddQueryAPWithoutMatch) {
    TrueAveragePrecision::QueryAPResult ap_result;
    ap_result.ap = 0.0;
    ap_result.has_potential_match = false;
    ap_result.rank_of_true_match = -1;
    ap_result.total_relevant = 0;
    
    metrics.addQueryAP("test_scene", ap_result);
    
    EXPECT_EQ(metrics.total_queries_processed, 0);
    EXPECT_EQ(metrics.total_queries_excluded, 1);
    EXPECT_TRUE(metrics.ap_per_query.empty());
    EXPECT_TRUE(metrics.per_scene_ap["test_scene"].empty());
    EXPECT_EQ(metrics.per_scene_excluded["test_scene"], 1);
    ASSERT_EQ(metrics.ranks_per_query.size(), 1);
    EXPECT_EQ(metrics.ranks_per_query[0], -1);
}

TEST_F(ExperimentMetricsTest, CalculateMeanPrecisionBasic) {
    metrics.addImageResult("scene1", 0.8, 20, 40);
    metrics.addImageResult("scene1", 0.6, 30, 60);
    metrics.addImageResult("scene2", 0.9, 10, 20);
    
    metrics.calculateMeanPrecision();
    
    // Mean precision: (0.8 + 0.6 + 0.9) / 3 = 0.7667
    EXPECT_NEAR(metrics.mean_precision, 0.7667, 1e-4);
    
    // Legacy macro precision: ((0.8 + 0.6)/2 + 0.9)/2 = (0.7 + 0.9)/2 = 0.8
    EXPECT_NEAR(metrics.legacy_macro_precision_by_scene, 0.8, 1e-6);
}

TEST_F(ExperimentMetricsTest, CalculateMeanPrecisionWithTrueMAPData) {
    // Add some AP data
    TrueAveragePrecision::QueryAPResult ap1;
    ap1.ap = 1.0; ap1.has_potential_match = true; ap1.rank_of_true_match = 1;
    
    TrueAveragePrecision::QueryAPResult ap2;
    ap2.ap = 0.5; ap2.has_potential_match = true; ap2.rank_of_true_match = 2;
    
    TrueAveragePrecision::QueryAPResult ap3;
    ap3.ap = 0.8; ap3.has_potential_match = true; ap3.rank_of_true_match = 1;
    
    metrics.addQueryAP("scene1", ap1);
    metrics.addQueryAP("scene1", ap2);
    metrics.addQueryAP("scene2", ap3);
    
    metrics.calculateMeanPrecision();
    
    // True micro mAP: (1.0 + 0.5 + 0.8) / 3 = 0.7667
    EXPECT_NEAR(metrics.true_map_micro, 0.7667, 1e-4);
    
    // True macro mAP: ((1.0 + 0.5)/2 + 0.8)/2 = (0.75 + 0.8)/2 = 0.775
    EXPECT_NEAR(metrics.true_map_macro_by_scene, 0.775, 1e-6);
}

TEST_F(ExperimentMetricsTest, CalculatePrecisionAtK) {
    // Add rank data for P@K/R@K calculation
    metrics.ranks_per_query = {1, 3, 1, 5, -1, 2, 10, 1};  // 7 valid ranks, 1 R=0
    
    metrics.calculateMeanPrecision();
    
    // Count hits at different K values (valid ranks only, ignore -1)
    // Ranks: [1, 3, 1, 5, 2, 10, 1] (7 valid queries)
    // Hits@1: 3 queries (ranks 1, 1, 1)
    // Hits@5: 6 queries (ranks 1, 3, 1, 5, 2, 1) - excludes rank 10
    // Hits@10: 7 queries (all valid ranks)
    
    EXPECT_NEAR(metrics.precision_at_1, 3.0/7.0, 1e-6);   // 3/7
    EXPECT_NEAR(metrics.precision_at_5, 6.0/7.0, 1e-6);   // 6/7  
    EXPECT_NEAR(metrics.precision_at_10, 7.0/7.0, 1e-6);  // 7/7 = 1.0
    
    // For R=1, Precision@K = Recall@K
    EXPECT_DOUBLE_EQ(metrics.recall_at_1, metrics.precision_at_1);
    EXPECT_DOUBLE_EQ(metrics.recall_at_5, metrics.precision_at_5);
    EXPECT_DOUBLE_EQ(metrics.recall_at_10, metrics.precision_at_10);
}

TEST_F(ExperimentMetricsTest, CalculateMeanPrecisionEmpty) {
    metrics.calculateMeanPrecision();
    
    EXPECT_EQ(metrics.mean_precision, 0.0);
    EXPECT_EQ(metrics.legacy_macro_precision_by_scene, 0.0);
    EXPECT_EQ(metrics.true_map_micro, 0.0);
    EXPECT_EQ(metrics.true_map_macro_by_scene, 0.0);
    EXPECT_EQ(metrics.precision_at_1, 0.0);
    EXPECT_EQ(metrics.precision_at_5, 0.0);
    EXPECT_EQ(metrics.precision_at_10, 0.0);
}

TEST_F(ExperimentMetricsTest, MergeBasic) {
    // First metrics
    metrics.addImageResult("scene1", 0.8, 20, 40);
    metrics.total_queries_processed = 5;
    
    // Second metrics
    ExperimentMetrics other_metrics;
    other_metrics.addImageResult("scene2", 0.6, 30, 60);
    other_metrics.total_queries_processed = 3;
    
    metrics.merge(other_metrics);
    
    EXPECT_EQ(metrics.total_images_processed, 2);  // 1 + 1
    EXPECT_EQ(metrics.total_matches, 50);          // 20 + 30
    EXPECT_EQ(metrics.total_keypoints, 100);       // 40 + 60
    EXPECT_EQ(metrics.total_queries_processed, 8); // 5 + 3
    
    // Check precision vectors merged
    ASSERT_EQ(metrics.precisions_per_image.size(), 2);
    EXPECT_DOUBLE_EQ(metrics.precisions_per_image[0], 0.8);
    EXPECT_DOUBLE_EQ(metrics.precisions_per_image[1], 0.6);
    
    // Check per-scene data
    EXPECT_EQ(metrics.per_scene_precisions.count("scene1"), 1);
    EXPECT_EQ(metrics.per_scene_precisions.count("scene2"), 1);
}

TEST_F(ExperimentMetricsTest, MergeWithErrors) {
    metrics.success = true;
    
    ExperimentMetrics failed_metrics;
    failed_metrics.success = false;
    failed_metrics.error_message = "Test error";
    
    metrics.merge(failed_metrics);
    
    EXPECT_FALSE(metrics.success);
    EXPECT_EQ(metrics.error_message, "Test error");
}

TEST_F(ExperimentMetricsTest, MergeWithMultipleErrors) {
    metrics.success = false;
    metrics.error_message = "First error";
    
    ExperimentMetrics other_failed;
    other_failed.success = false;
    other_failed.error_message = "Second error";
    
    metrics.merge(other_failed);
    
    EXPECT_FALSE(metrics.success);
    EXPECT_EQ(metrics.error_message, "First error; Second error");
}

TEST_F(ExperimentMetricsTest, MergeSameScene) {
    // Add to same scene from different metrics objects
    metrics.addImageResult("shared_scene", 0.7, 10, 20);
    
    ExperimentMetrics other_metrics;
    other_metrics.addImageResult("shared_scene", 0.9, 15, 30);
    
    metrics.merge(other_metrics);
    
    // Should have merged the scene data
    ASSERT_EQ(metrics.per_scene_precisions["shared_scene"].size(), 2);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["shared_scene"][0], 0.7);
    EXPECT_DOUBLE_EQ(metrics.per_scene_precisions["shared_scene"][1], 0.9);
    EXPECT_EQ(metrics.per_scene_matches["shared_scene"], 25);      // 10 + 15
    EXPECT_EQ(metrics.per_scene_keypoints["shared_scene"], 50);    // 20 + 30
    EXPECT_EQ(metrics.per_scene_image_count["shared_scene"], 2);   // 1 + 1
}

TEST_F(ExperimentMetricsTest, GetSceneAveragePrecision) {
    metrics.addImageResult("test_scene", 0.8, 10, 20);
    metrics.addImageResult("test_scene", 0.6, 15, 30);
    
    double avg_precision = metrics.getSceneAveragePrecision("test_scene");
    EXPECT_NEAR(avg_precision, 0.7, 1e-6);  // (0.8 + 0.6) / 2
    
    // Non-existent scene
    double missing_precision = metrics.getSceneAveragePrecision("missing_scene");
    EXPECT_EQ(missing_precision, 0.0);
}

TEST_F(ExperimentMetricsTest, GetSceneNames) {
    metrics.addImageResult("scene_a", 0.5, 10, 20);
    metrics.addImageResult("scene_b", 0.8, 15, 30);
    metrics.addImageResult("scene_a", 0.7, 5, 10);
    
    std::vector<std::string> scene_names = metrics.getSceneNames();
    
    EXPECT_EQ(scene_names.size(), 2);
    // Order is not guaranteed, so check both are present
    EXPECT_TRUE(std::find(scene_names.begin(), scene_names.end(), "scene_a") != scene_names.end());
    EXPECT_TRUE(std::find(scene_names.begin(), scene_names.end(), "scene_b") != scene_names.end());
}

TEST_F(ExperimentMetricsTest, CreateError) {
    ExperimentMetrics error_metrics = ExperimentMetrics::createError("Test error message");
    
    EXPECT_FALSE(error_metrics.success);
    EXPECT_EQ(error_metrics.error_message, "Test error message");
    EXPECT_EQ(error_metrics.total_images_processed, 0);
    EXPECT_EQ(error_metrics.mean_precision, 0.0);
}

TEST_F(ExperimentMetricsTest, CreateSuccess) {
    ExperimentMetrics success_metrics = ExperimentMetrics::createSuccess();
    
    EXPECT_TRUE(success_metrics.success);
    EXPECT_TRUE(success_metrics.error_message.empty());
    EXPECT_EQ(success_metrics.total_images_processed, 0);
    EXPECT_EQ(success_metrics.mean_precision, 0.0);
}

TEST_F(ExperimentMetricsTest, TrueMAPIncludingZeros) {
    // Add some queries with matches and some R=0 queries
    TrueAveragePrecision::QueryAPResult matched_query;
    matched_query.ap = 0.8; 
    matched_query.has_potential_match = true;
    
    TrueAveragePrecision::QueryAPResult excluded_query;
    excluded_query.ap = 0.0;
    excluded_query.has_potential_match = false;
    
    metrics.addQueryAP("scene1", matched_query);  // Processed
    metrics.addQueryAP("scene1", excluded_query); // Excluded (R=0)
    metrics.addQueryAP("scene2", matched_query);  // Processed
    
    metrics.calculateMeanPrecision();
    
    // Regular mAP: (0.8 + 0.8) / 2 = 0.8 (excludes R=0 queries)
    EXPECT_NEAR(metrics.true_map_micro, 0.8, 1e-6);
    
    // Including zeros mAP: (0.8 + 0.0 + 0.8) / 3 = 0.533 (includes R=0 as AP=0)
    EXPECT_NEAR(metrics.true_map_micro_including_zeros, 0.533333, 1e-6);
}