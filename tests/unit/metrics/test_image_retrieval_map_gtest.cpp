#include <gtest/gtest.h>
#include "src/core/metrics/ExperimentMetrics.hpp"

TEST(ExperimentMetricsImageRetrievalTest, ComputesAverageFromQueries) {
    ExperimentMetrics metrics;

    metrics.addImageRetrievalAP("scene_a", 1.0);
    metrics.addImageRetrievalAP("scene_a", 0.5);
    metrics.addImageRetrievalAP("scene_b", 0.25);

    metrics.calculateMeanPrecision();

    EXPECT_NEAR(metrics.image_retrieval_map, (1.0 + 0.5 + 0.25) / 3.0, 1e-6);
    EXPECT_EQ(metrics.image_retrieval_queries, 3);
    ASSERT_EQ(metrics.image_retrieval_ap_per_scene["scene_a"].size(), 2);
    ASSERT_EQ(metrics.image_retrieval_ap_per_scene["scene_b"].size(), 1);
}

TEST(ExperimentMetricsImageRetrievalTest, DefaultsToSentinelWhenNoQueries) {
    ExperimentMetrics metrics;
    metrics.calculateMeanPrecision();

    EXPECT_DOUBLE_EQ(metrics.image_retrieval_map, -1.0);
    EXPECT_EQ(metrics.image_retrieval_queries, 0);
    EXPECT_TRUE(metrics.image_retrieval_ap_per_query.empty());
}
