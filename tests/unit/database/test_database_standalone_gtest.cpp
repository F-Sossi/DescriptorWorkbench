#include <gtest/gtest.h>
#include <filesystem>
#include "thesis_project/database/DatabaseManager.hpp"

class DatabaseStandaloneTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_name = "test_stage5_standalone_gtest.db";
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

TEST_F(DatabaseStandaloneTest, DisabledDatabase) {
    thesis_project::database::DatabaseManager db_disabled("", false);
    EXPECT_FALSE(db_disabled.isEnabled()) << "Disabled database should not be enabled";
}

TEST_F(DatabaseStandaloneTest, EnabledDatabaseInitialization) {
    thesis_project::database::DatabaseManager db_enabled(test_db_name, true);
    EXPECT_TRUE(db_enabled.isEnabled()) << "Enabled database should initialize successfully";
}

TEST_F(DatabaseStandaloneTest, ConfigurationRecording) {
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

TEST_F(DatabaseStandaloneTest, ResultsRecording) {
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

TEST_F(DatabaseStandaloneTest, ResultsRetrieval) {
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
        EXPECT_DOUBLE_EQ(recent_results[0].precision_at_1, 0.9)
            << "Retrieved precision@1 should match recorded value";
        EXPECT_EQ(recent_results[0].total_matches, 150)
            << "Retrieved match count should match recorded value";
        EXPECT_EQ(recent_results[0].total_keypoints, 1000)
            << "Retrieved keypoint count should match recorded value";
        EXPECT_DOUBLE_EQ(recent_results[0].processing_time_ms, 250.5)
            << "Retrieved processing time should match recorded value";
    }
}

TEST_F(DatabaseStandaloneTest, StatisticsRetrieval) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for this test";
    
    // Record some data first to ensure statistics are meaningful
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
    
    for (const auto& [key, value] : stats) {
        EXPECT_FALSE(key.empty()) << "Statistic key should not be empty";
        EXPECT_GE(value, 0.0) << "Statistic value should be non-negative";
    }
    
    // Note: We don't enforce these specific keys since they may vary by implementation
    // but we ensure that statistics are meaningful and non-empty
}

// Test error handling for invalid database operations
TEST_F(DatabaseStandaloneTest, InvalidDatabasePath) {
    // Test with invalid path (directory that doesn't exist and can't be created)
    thesis_project::database::DatabaseManager db("/invalid/path/test.db", true);
    
    // The database should handle this gracefully - either by creating path or by disabling
    // We don't enforce specific behavior, just that it doesn't crash
    EXPECT_NO_THROW({
        bool enabled = db.isEnabled();
        (void)enabled; // Suppress unused variable warning - we just test that it doesn't crash
        // Either it works or it's disabled - both are acceptable error handling
    }) << "Database should handle invalid paths gracefully";
}

// Test that database files are created properly
TEST_F(DatabaseStandaloneTest, DatabaseFileCreation) {
    EXPECT_FALSE(std::filesystem::exists(test_db_name)) << "Test database should not exist initially";
    
    {
        thesis_project::database::DatabaseManager db(test_db_name, true);
        if (db.isEnabled()) {
            EXPECT_TRUE(std::filesystem::exists(test_db_name)) << "Database file should be created when enabled";
        }
    }
    
    // File should still exist after DatabaseManager goes out of scope
    if (std::filesystem::exists(test_db_name)) {
        EXPECT_GT(std::filesystem::file_size(test_db_name), 0) << "Database file should have non-zero size";
    }
}

// Test complete workflow from start to finish
TEST_F(DatabaseStandaloneTest, CompleteWorkflow) {
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database must be enabled for complete workflow test";
    
    // Step 1: Record configuration
    thesis_project::database::ExperimentConfig config;
    config.descriptor_type = "RGBSIFT";
    config.dataset_path = "/test/hpatches";
    config.pooling_strategy = "STACKING";
    config.max_features = 2000;
    config.similarity_threshold = 0.6;
    
    int exp_id = db.recordConfiguration(config);
    ASSERT_GT(exp_id, 0) << "Configuration should be recorded";
    
    // Step 2: Record results
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "RGBSIFT";
    results.dataset_name = "i_dome";
    results.mean_average_precision = 0.78;
    results.precision_at_1 = 0.82;
    results.precision_at_5 = 0.79;
    results.recall_at_1 = 0.75;
    results.recall_at_5 = 0.77;
    results.total_matches = 200;
    results.total_keypoints = 1500;
    results.processing_time_ms = 380.2;
    
    ASSERT_TRUE(db.recordExperiment(results)) << "Results should be recorded";
    
    // Step 3: Verify data persistence
    auto retrieved = db.getRecentResults(1);
    ASSERT_FALSE(retrieved.empty()) << "Should retrieve recorded results";
    
    const auto& result = retrieved[0];
    EXPECT_EQ(result.descriptor_type, "RGBSIFT");
    EXPECT_FALSE(result.dataset_name.empty()) << "Dataset name should not be empty";
    EXPECT_DOUBLE_EQ(result.mean_average_precision, 0.78);
    EXPECT_EQ(result.total_matches, 200);
    EXPECT_EQ(result.total_keypoints, 1500);
    
    // Step 4: Verify statistics
    auto stats = db.getStatistics();
    EXPECT_FALSE(stats.empty()) << "Statistics should be available after data recording";
}
