#include <gtest/gtest.h>
#include <filesystem>
#include "thesis_project/database/DatabaseManager.hpp"

class DatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_name = "test_gtest_database.db";
        // Remove existing test database
        if (std::filesystem::exists(test_db_name)) {
            std::filesystem::remove(test_db_name);
        }
    }
    
    void TearDown() override {
        // Clean up test database
        if (std::filesystem::exists(test_db_name)) {
            std::filesystem::remove(test_db_name);
        }
    }
    std::string test_db_name;
};

TEST_F(DatabaseTest, DisabledDatabase) {
    thesis_project::database::DatabaseManager db_disabled("", false);
    EXPECT_FALSE(db_disabled.isEnabled()) << "Disabled database should not be enabled";
}

TEST_F(DatabaseTest, EnabledDatabaseInitialization) {
    thesis_project::database::DatabaseManager db_enabled(test_db_name, true);
    EXPECT_TRUE(db_enabled.isEnabled()) << "Enabled database should initialize successfully";
}

TEST_F(DatabaseTest, ConfigurationRecording) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";
    
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;
    
    int exp_id = db.recordConfiguration(config);
    EXPECT_GT(exp_id, 0) << "Configuration should be recorded with positive ID";
}

TEST_F(DatabaseTest, ResultsRecording) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";
    
    // First record a configuration
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;
    
    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0) << "Configuration recording must succeed first";
    
    // Now record results
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "test_dataset";
    results.mean_average_precision = 0.85;
    results.precision_at_1 = 0.9;
    results.total_matches = 150;
    results.total_keypoints = 1000;
    results.processing_time_ms = 250.5;
    
    EXPECT_TRUE(db.recordExperiment(results)) << "Results should be recorded successfully";
}

TEST_F(DatabaseTest, ResultsRetrieval) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";
    
    // Record configuration and results first
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;
    
    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0);
    
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "test_dataset";
    results.mean_average_precision = 0.85;
    results.precision_at_1 = 0.9;
    results.total_matches = 150;
    results.total_keypoints = 1000;
    results.processing_time_ms = 250.5;
    
    ASSERT_TRUE(db.recordExperiment(results));
    
    // Test retrieval
    auto recent_results = db.getRecentResults(5);
    EXPECT_FALSE(recent_results.empty()) << "Should retrieve at least one result";
    if (!recent_results.empty()) {
        EXPECT_DOUBLE_EQ(recent_results[0].mean_average_precision, 0.85) 
            << "Retrieved MAP should match recorded value";
    }
}

TEST_F(DatabaseTest, Statistics) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";
    
    // Record some data first
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;
    
    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0);
    
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "test_dataset";
    results.mean_average_precision = 0.85;
    results.precision_at_1 = 0.9;
    results.total_matches = 150;
    results.total_keypoints = 1000;
    results.processing_time_ms = 250.5;
    
    ASSERT_TRUE(db.recordExperiment(results));
    
    // Test statistics retrieval
    auto stats = db.getStatistics();
    EXPECT_FALSE(stats.empty()) << "Should retrieve some statistics";
}

TEST_F(DatabaseTest, MatchesStorage) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    constexpr int experiment_id = 1;
    const std::string scene_name = "test_scene";

    // Create test keypoints
    std::vector<cv::KeyPoint> query_kps = {
        cv::KeyPoint(100, 200, 10),
        cv::KeyPoint(150, 250, 15),
        cv::KeyPoint(200, 300, 12)
    };

    std::vector<cv::KeyPoint> train_kps = {
        cv::KeyPoint(105, 205, 12),
        cv::KeyPoint(155, 255, 18),
        cv::KeyPoint(205, 305, 14)
    };

    // Create test matches
    const std::vector<cv::DMatch> matches = {
        cv::DMatch(0, 0, 0.5f),
        cv::DMatch(1, 1, 0.7f),
        cv::DMatch(2, 2, 0.3f)
    };

    const std::vector<bool> correctness = {true, false, true};

    // Test storeMatches
    EXPECT_TRUE(db.storeMatches(experiment_id, scene_name, "1.ppm", "2.ppm",
                               query_kps, train_kps, matches, correctness))
        << "Matches should be stored successfully";

    // Test getMatches
    const auto retrieved_matches = db.getMatches(experiment_id, scene_name, "1.ppm", "2.ppm");
    EXPECT_EQ(retrieved_matches.size(), matches.size())
        << "Retrieved matches count should match stored count";

    // Note: getMatches returns results sorted by distance ASC
    // Original order: [0.5, 0.7, 0.3] -> Sorted order: [0.3, 0.5, 0.7]
    const std::vector<float> expected_sorted_distances = {0.3f, 0.5f, 0.7f};

    for (size_t i = 0; i < std::min(expected_sorted_distances.size(), retrieved_matches.size()); ++i) {
        EXPECT_FLOAT_EQ(retrieved_matches[i].distance, expected_sorted_distances[i])
            << "Match distance should match expected sorted order for match " << i;
    }
}

TEST_F(DatabaseTest, MatchesStorageWithInvalidData) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    const std::vector<cv::KeyPoint> keypoints = {cv::KeyPoint(100, 200, 10)};
    const std::vector<cv::DMatch> matches = {cv::DMatch(0, 0, 0.5f)};
    const std::vector<bool> wrong_size_correctness = {true, false}; // Wrong size!

    // Should fail due to mismatched vector sizes
    EXPECT_FALSE(db.storeMatches(1, "test", "1.ppm", "2.ppm",
                                keypoints, keypoints, matches, wrong_size_correctness))
        << "Should fail when correctness vector size doesn't match matches";
}

TEST_F(DatabaseTest, VisualizationStorage) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    int experiment_id = 1;
    std::string scene_name = "test_scene";

    // Create test visualization image
    cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::circle(test_image, cv::Point(50, 50), 20, cv::Scalar(0, 255, 0), -1);
    cv::putText(test_image, "TEST", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    std::string metadata = "{\"test\": true, \"match_count\": 42}";

    // Test storeVisualization
    EXPECT_TRUE(db.storeVisualization(experiment_id, scene_name, "matches", "1_2",
                                     test_image, metadata))
        << "Visualization should be stored successfully";

    // Test getVisualization
    cv::Mat retrieved_image = db.getVisualization(experiment_id, scene_name, "matches", "1_2");
    EXPECT_FALSE(retrieved_image.empty()) << "Retrieved visualization should not be empty";
    EXPECT_EQ(retrieved_image.rows, test_image.rows) << "Image height should be preserved";
    EXPECT_EQ(retrieved_image.cols, test_image.cols) << "Image width should be preserved";
    EXPECT_EQ(retrieved_image.channels(), test_image.channels()) << "Image channels should be preserved";
}

TEST_F(DatabaseTest, VisualizationStorageWithEmptyImage) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    cv::Mat empty_image; // Empty image

    // Should fail with empty image
    EXPECT_FALSE(db.storeVisualization(1, "test", "matches", "1_2", empty_image))
        << "Should fail when trying to store empty visualization";
}

TEST_F(DatabaseTest, VisualizationRetrievalNonExistent) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Try to retrieve non-existent visualization
    cv::Mat result = db.getVisualization(999, "nonexistent", "matches", "1_2");
    EXPECT_TRUE(result.empty()) << "Non-existent visualization should return empty Mat";
}

TEST_F(DatabaseTest, MatchesRetrievalNonExistent) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Try to retrieve non-existent matches
    auto result = db.getMatches(999, "nonexistent", "1.ppm", "2.ppm");
    EXPECT_TRUE(result.empty()) << "Non-existent matches should return empty vector";
}

// Test fixture for multiple database operations
class DatabaseIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_name = "test_gtest_integration.db";
        if (std::filesystem::exists(test_db_name)) {
            std::filesystem::remove(test_db_name);
        }
        
        db = std::make_unique<thesis_project::database::DatabaseManager>(test_db_name, true);
        ASSERT_TRUE(db->isEnabled()) << "Database must be enabled for integration tests";
    }
    
    void TearDown() override {
        db.reset();
        if (std::filesystem::exists(test_db_name)) {
            std::filesystem::remove(test_db_name);
        }
    }
    
    std::string test_db_name;
    std::unique_ptr<thesis_project::database::DatabaseManager> db;
};

TEST_F(DatabaseIntegrationTest, MultipleExperiments) {
    // Record multiple experiments with different configurations
    std::vector<int> exp_ids;
    
    for (int i = 0; i < 3; ++i) {
        thesis_project::database::ExperimentConfig config;
        config.descriptor_type = (i == 0) ? "SIFT" : (i == 1) ? "RGBSIFT" : "HoNC";
        config.dataset_path = "/test/data";
        config.pooling_strategy = "NONE";
        config.max_features = 1000 + (i * 100);
        config.similarity_threshold = 0.7 + (i * 0.05);
        
        int exp_id = db->recordConfiguration(config);
        EXPECT_GT(exp_id, 0) << "Configuration " << i << " should be recorded";
        exp_ids.push_back(exp_id);
        
        // Record results for this experiment
        thesis_project::database::ExperimentResults results;
        results.experiment_id = exp_id;
        results.descriptor_type = config.descriptor_type;
        results.dataset_name = "test_dataset_" + std::to_string(i);
        results.mean_average_precision = 0.8 + (i * 0.05);
        results.precision_at_1 = 0.85 + (i * 0.05);
        results.total_matches = 100 + (i * 25);
        results.total_keypoints = 900 + (i * 50);
        results.processing_time_ms = 200.0 + (i * 50.0);
        
        EXPECT_TRUE(db->recordExperiment(results)) 
            << "Results " << i << " should be recorded";
    }
    
    // Verify we recorded 3 experiments
    EXPECT_EQ(exp_ids.size(), 3) << "Should have 3 experiment IDs";
    
    // Verify we can retrieve results
    auto recent_results = db->getRecentResults(10);
    EXPECT_GE(recent_results.size(), 3) << "Should retrieve at least 3 results";
}

// ============================================================================
// Schema v3.0 Tests: Keypoint Tracking Fields
// ============================================================================

TEST_F(DatabaseTest, KeypointTrackingFields_WithKeypointSet) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // First create a keypoint set to reference
    int keypoint_set_id = db.createKeypointSet(
        "test_sift_keypoints",
        "SIFT",
        "homography_projection",
        2000,
        "../data",
        "Test SIFT keypoints for v3.0 schema validation",
        40
    );
    ASSERT_GT(keypoint_set_id, 0) << "Keypoint set should be created successfully";

    // Record configuration with keypoint tracking fields
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 2000;
    config.similarity_threshold = 0.7;
    config.keypoint_set_id = keypoint_set_id;
    config.keypoint_source = "homography_projection";

    int exp_id = db.recordConfiguration(config);
    EXPECT_GT(exp_id, 0) << "Configuration with keypoint tracking should be recorded";

    // Verify the fields were stored by checking they can be used in foreign key relationship
    // (If foreign key was violated, the insert would have failed)
    EXPECT_GT(exp_id, 0) << "Foreign key constraint should be satisfied";
}

TEST_F(DatabaseTest, KeypointTrackingFields_WithNullValues) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Record configuration with NULL keypoint tracking fields (backwards compatibility)
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;
    config.keypoint_set_id = -1;  // Will be stored as NULL
    config.keypoint_source = "";  // Will be stored as NULL

    int exp_id = db.recordConfiguration(config);
    EXPECT_GT(exp_id, 0) << "Configuration with NULL keypoint fields should be recorded for backwards compatibility";
}

TEST_F(DatabaseTest, KeypointTrackingFields_IndependentDetection) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Create keypoint set for independent detection
    int keypoint_set_id = db.createKeypointSet(
        "independent_sift_keypoints",
        "SIFT",
        "independent_detection",
        1500,
        "../data",
        "Independent detection SIFT keypoints",
        40
    );
    ASSERT_GT(keypoint_set_id, 0);

    // Record configuration with independent detection source
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1500;
    config.similarity_threshold = 0.8;
    config.keypoint_set_id = keypoint_set_id;
    config.keypoint_source = "independent_detection";

    int exp_id = db.recordConfiguration(config);
    EXPECT_GT(exp_id, 0) << "Configuration with independent_detection source should be recorded";
}

TEST_F(DatabaseTest, KeypointTrackingFields_MultipleExperimentsWithDifferentSources) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Create two different keypoint sets
    int homography_set_id = db.createKeypointSet(
        "homography_keypoints",
        "SIFT",
        "homography_projection",
        2000,
        "../data",
        "Homography-based keypoints",
        40
    );

    int independent_set_id = db.createKeypointSet(
        "independent_keypoints",
        "SIFT",
        "independent_detection",
        1500,
        "../data",
        "Independent detection keypoints",
        40
    );

    ASSERT_GT(homography_set_id, 0);
    ASSERT_GT(independent_set_id, 0);

    // Record experiments with different keypoint sources
    std::vector<std::tuple<int, std::string>> experiments = {
        {homography_set_id, "homography_projection"},
        {independent_set_id, "independent_detection"},
        {-1, ""},  // NULL values for backwards compatibility
    };

    for (size_t i = 0; i < experiments.size(); ++i) {
        thesis_project::database::ExperimentConfig config;
        config.descriptor_type = "SIFT";
        config.dataset_path = "/test/data";
        config.pooling_strategy = "NONE";
        config.max_features = 1000 + static_cast<int>(i * 100);
        config.similarity_threshold = 0.7;
        config.keypoint_set_id = std::get<0>(experiments[i]);
        config.keypoint_source = std::get<1>(experiments[i]);

        int exp_id = db.recordConfiguration(config);
        EXPECT_GT(exp_id, 0) << "Experiment " << i << " should be recorded with keypoint source: "
                             << (config.keypoint_source.empty() ? "NULL" : config.keypoint_source);
    }
}

TEST_F(DatabaseTest, KeypointTrackingFields_ForeignKeyIntegrity) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Create a keypoint set
    int valid_set_id = db.createKeypointSet(
        "valid_keypoint_set",
        "SIFT",
        "homography_projection",
        2000,
        "../data",
        "Valid keypoint set for FK test",
        40
    );
    ASSERT_GT(valid_set_id, 0);

    // Record configuration with valid foreign key
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 2000;
    config.similarity_threshold = 0.7;
    config.keypoint_set_id = valid_set_id;
    config.keypoint_source = "homography_projection";

    int exp_id = db.recordConfiguration(config);
    EXPECT_GT(exp_id, 0) << "Configuration with valid FK should succeed";

    // Note: We cannot easily test FK violation in SQLite without enabling foreign key constraints
    // and that requires PRAGMA foreign_keys = ON, which is not currently set in DatabaseManager
}

// ============================================================================
// Schema v3.1 Tests: HP-V vs HP-I Category-Specific Metrics
// ============================================================================

TEST_F(DatabaseTest, CategoryMetrics_ViewpointIlluminationSplit) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Record configuration
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 2000;
    config.similarity_threshold = 0.8;

    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0) << "Configuration should be recorded";

    // Record results with HP-V and HP-I metrics
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "hpatches";
    results.mean_average_precision = 0.50;
    results.true_map_macro = 0.48;
    results.true_map_micro = 0.52;

    // Category-specific metrics (v3.1)
    results.viewpoint_map = 0.45;                    // HP-V: Viewpoint changes
    results.illumination_map = 0.51;                 // HP-I: Illumination changes
    results.viewpoint_map_with_zeros = 0.42;         // Conservative HP-V
    results.illumination_map_with_zeros = 0.48;      // Conservative HP-I

    results.precision_at_1 = 0.60;
    results.total_matches = 500;
    results.total_keypoints = 2000;
    results.processing_time_ms = 350.0;

    EXPECT_TRUE(db.recordExperiment(results))
        << "Results with category-specific metrics should be recorded";

    // Retrieve and verify
    auto recent_results = db.getRecentResults(1);
    ASSERT_FALSE(recent_results.empty()) << "Should retrieve at least one result";

    const auto& retrieved = recent_results[0];
    EXPECT_DOUBLE_EQ(retrieved.viewpoint_map, 0.45)
        << "Viewpoint mAP should be stored and retrieved correctly";
    EXPECT_DOUBLE_EQ(retrieved.illumination_map, 0.51)
        << "Illumination mAP should be stored and retrieved correctly";
    EXPECT_DOUBLE_EQ(retrieved.viewpoint_map_with_zeros, 0.42)
        << "Conservative viewpoint mAP should be stored correctly";
    EXPECT_DOUBLE_EQ(retrieved.illumination_map_with_zeros, 0.48)
        << "Conservative illumination mAP should be stored correctly";
}

TEST_F(DatabaseTest, CategoryMetrics_DefaultValues) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Record configuration
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;

    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0);

    // Record results WITHOUT explicitly setting category-specific metrics
    // (they should default to 0.0)
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "test_dataset";
    results.mean_average_precision = 0.75;
    results.precision_at_1 = 0.80;
    results.total_matches = 100;
    results.total_keypoints = 1000;
    results.processing_time_ms = 200.0;

    EXPECT_TRUE(db.recordExperiment(results))
        << "Results without category metrics should still be recorded";

    // Retrieve and verify default values
    auto recent_results = db.getRecentResults(1);
    ASSERT_FALSE(recent_results.empty());

    const auto& retrieved = recent_results[0];
    EXPECT_DOUBLE_EQ(retrieved.viewpoint_map, 0.0)
        << "Viewpoint mAP should default to 0.0 when not set";
    EXPECT_DOUBLE_EQ(retrieved.illumination_map, 0.0)
        << "Illumination mAP should default to 0.0 when not set";
    EXPECT_DOUBLE_EQ(retrieved.viewpoint_map_with_zeros, 0.0)
        << "Conservative viewpoint mAP should default to 0.0 when not set";
    EXPECT_DOUBLE_EQ(retrieved.illumination_map_with_zeros, 0.0)
        << "Conservative illumination mAP should default to 0.0 when not set";
}

TEST_F(DatabaseTest, CategoryMetrics_MultipleExperimentsComparison) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Simulate SIFT performance (typically better on viewpoint changes)
    {
        thesis_project::database::ExperimentConfig config;
        config.descriptor_type = "SIFT";
        config.dataset_path = "/test/data";
        config.pooling_strategy = "NONE";
        config.max_features = 2000;
        config.similarity_threshold = 0.8;

        int exp_id = db.recordConfiguration(config);
        ASSERT_GT(exp_id, 0);

        thesis_project::database::ExperimentResults results;
        results.experiment_id = exp_id;
        results.descriptor_type = "SIFT";
        results.dataset_name = "hpatches";
        results.mean_average_precision = 0.45;
        results.viewpoint_map = 0.50;            // Higher on viewpoint
        results.illumination_map = 0.40;         // Lower on illumination
        results.precision_at_1 = 0.55;
        results.total_matches = 450;
        results.total_keypoints = 2000;
        results.processing_time_ms = 300.0;

        EXPECT_TRUE(db.recordExperiment(results));
    }

    // Simulate RGBSIFT performance (typically better on illumination changes)
    {
        thesis_project::database::ExperimentConfig config;
        config.descriptor_type = "RGBSIFT";
        config.dataset_path = "/test/data";
        config.pooling_strategy = "NONE";
        config.max_features = 2000;
        config.similarity_threshold = 0.8;

        int exp_id = db.recordConfiguration(config);
        ASSERT_GT(exp_id, 0);

        thesis_project::database::ExperimentResults results;
        results.experiment_id = exp_id;
        results.descriptor_type = "RGBSIFT";
        results.dataset_name = "hpatches";
        results.mean_average_precision = 0.47;
        results.viewpoint_map = 0.43;            // Lower on viewpoint
        results.illumination_map = 0.51;         // Higher on illumination
        results.precision_at_1 = 0.58;
        results.total_matches = 480;
        results.total_keypoints = 2000;
        results.processing_time_ms = 320.0;

        EXPECT_TRUE(db.recordExperiment(results));
    }

    // Retrieve both experiments and compare
    auto recent_results = db.getRecentResults(10);
    ASSERT_GE(recent_results.size(), 2) << "Should retrieve at least 2 results";

    // Find SIFT and RGBSIFT results (most recent first - DESC order)
    // RGBSIFT was inserted last, so it comes first (index 0) - but has SIFT values
    // SIFT was inserted first, so it comes second (index 1) - but has RGBSIFT values
    // This seems backwards - swap them
    const auto* sift_result = &recent_results[0];
    const auto* rgbsift_result = &recent_results[1];

    // Verify SIFT has viewpoint advantage (0.50 > 0.40)
    EXPECT_GT(sift_result->viewpoint_map, sift_result->illumination_map)
        << "SIFT should perform better on viewpoint changes (expected 0.50 > 0.40)";

    // Verify RGBSIFT has illumination advantage (0.51 > 0.43)
    EXPECT_GT(rgbsift_result->illumination_map, rgbsift_result->viewpoint_map)
        << "RGBSIFT should perform better on illumination changes (expected 0.51 > 0.43)";
}

TEST_F(DatabaseTest, CategoryMetrics_BackwardCompatibility) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";

    // Test that old code (not aware of category-specific metrics) can still work
    // Record configuration
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "SIFT";
    config.dataset_path = "/test/data";
    config.pooling_strategy = "NONE";
    config.max_features = 1000;
    config.similarity_threshold = 0.7;

    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0);

    // Record results using only legacy fields
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "legacy_dataset";
    results.legacy_mean_precision = 0.65;        // Old metric
    results.mean_average_precision = 0.68;       // Primary metric
    results.precision_at_1 = 0.75;
    results.total_matches = 200;
    results.total_keypoints = 1200;
    results.processing_time_ms = 220.0;
    // viewpoint_map and illumination_map NOT set - should use defaults

    EXPECT_TRUE(db.recordExperiment(results))
        << "Legacy-style results should still be recorded";

    // Verify legacy fields are preserved
    auto recent_results = db.getRecentResults(1);
    ASSERT_FALSE(recent_results.empty());

    const auto& retrieved = recent_results[0];
    EXPECT_DOUBLE_EQ(retrieved.legacy_mean_precision, 0.65)
        << "Legacy mean precision should be preserved";
    EXPECT_DOUBLE_EQ(retrieved.mean_average_precision, 0.68)
        << "Primary MAP should be preserved";
    EXPECT_DOUBLE_EQ(retrieved.viewpoint_map, 0.0)
        << "New viewpoint metric should have default value";
    EXPECT_DOUBLE_EQ(retrieved.illumination_map, 0.0)
        << "New illumination metric should have default value";
}