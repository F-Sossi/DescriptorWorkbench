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