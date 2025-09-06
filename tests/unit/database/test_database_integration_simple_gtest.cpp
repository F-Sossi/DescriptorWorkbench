#include <gtest/gtest.h>
#include <filesystem>
#include "thesis_project/database/DatabaseManager.hpp"

// Simple mock of experiment_config for testing (no linking needed)
enum DescriptorType { DESCRIPTOR_SIFT, DESCRIPTOR_RGBSIFT, DESCRIPTOR_HoNC, DESCRIPTOR_vSIFT };
enum PoolingStrategy { NONE, STACKING, DOMAIN_SIZE_POOLING };
enum ImageType { COLOR, BW };
enum DescriptorColorSpace { D_COLOR, D_BW };

struct DescriptorOptions {
    DescriptorType descriptorType = DESCRIPTOR_SIFT;
    PoolingStrategy poolingStrategy = NONE;
    int normType = 2;
    ImageType imageType = COLOR;
    DescriptorColorSpace descriptorColorSpace = D_COLOR;
};

struct MockExperimentConfig {
    DescriptorOptions descriptorOptions;
    bool useMultiThreading = true;
    double matchThreshold = 0.05;
};

// Simple integration functions (no external dependencies)
namespace simple_integration {
    inline thesis_project::database::ExperimentConfig toDbConfig(const MockExperimentConfig& config) {
        thesis_project::database::ExperimentConfig db_config;
        
        // Convert descriptor type
        switch (config.descriptorOptions.descriptorType) {
            case DESCRIPTOR_SIFT:
                db_config.descriptor_type = "SIFT";
                break;
            case DESCRIPTOR_RGBSIFT:
                db_config.descriptor_type = "RGBSIFT";
                break;
            case DESCRIPTOR_HoNC:
                db_config.descriptor_type = "HoNC";
                break;
            case DESCRIPTOR_vSIFT:
                db_config.descriptor_type = "vSIFT";
                break;
        }
        
        db_config.dataset_path = "/test/data";
        
        // Convert pooling strategy
        if (config.descriptorOptions.poolingStrategy == STACKING) {
            db_config.pooling_strategy = "STACKING";
        } else if (config.descriptorOptions.poolingStrategy == DOMAIN_SIZE_POOLING) {
            db_config.pooling_strategy = "DOMAIN_SIZE_POOLING";
        } else {
            db_config.pooling_strategy = "NONE";
        }

        db_config.max_features = 1000;
        db_config.similarity_threshold = config.matchThreshold;

        // Add parameters
        db_config.parameters["normType"] = std::to_string(config.descriptorOptions.normType);
        db_config.parameters["useMultiThreading"] = config.useMultiThreading ? "true" : "false";
        db_config.parameters["imageType"] = (config.descriptorOptions.imageType == COLOR) ? "COLOR" : "BW";
        db_config.parameters["descriptorColorSpace"] = (config.descriptorOptions.descriptorColorSpace == D_COLOR) ? "D_COLOR" : "D_BW";

        return db_config;
    }
}

class DatabaseIntegrationSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_db_name = "test_integration_simple_gtest.db";
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

TEST_F(DatabaseIntegrationSimpleTest, MockConfigConversion) {
    // Create mock configuration
    MockExperimentConfig config;
    config.descriptorOptions.descriptorType = DESCRIPTOR_RGBSIFT;
    config.descriptorOptions.poolingStrategy = STACKING;
    config.descriptorOptions.normType = 2;
    config.descriptorOptions.imageType = COLOR;
    config.descriptorOptions.descriptorColorSpace = D_COLOR;
    config.useMultiThreading = true;
    config.matchThreshold = 0.05;

    // Convert to database format
    auto db_config = simple_integration::toDbConfig(config);

    // Verify conversion
    EXPECT_EQ(db_config.descriptor_type, "RGBSIFT") << "Descriptor type should be converted correctly";
    EXPECT_EQ(db_config.pooling_strategy, "STACKING") << "Pooling strategy should be converted correctly";
    EXPECT_EQ(db_config.max_features, 1000) << "Max features should be set correctly";
    EXPECT_DOUBLE_EQ(db_config.similarity_threshold, 0.05) << "Similarity threshold should be set correctly";
    
    // Verify parameters
    EXPECT_EQ(db_config.parameters.at("normType"), "2") << "NormType parameter should be converted";
    EXPECT_EQ(db_config.parameters.at("useMultiThreading"), "true") << "MultiThreading parameter should be converted";
    EXPECT_EQ(db_config.parameters.at("imageType"), "COLOR") << "ImageType parameter should be converted";
    EXPECT_EQ(db_config.parameters.at("descriptorColorSpace"), "D_COLOR") << "ColorSpace parameter should be converted";
}

TEST_F(DatabaseIntegrationSimpleTest, DatabaseIntegrationWorkflow) {
    // Create and convert configuration
    MockExperimentConfig config;
    config.descriptorOptions.descriptorType = DESCRIPTOR_RGBSIFT;
    config.descriptorOptions.poolingStrategy = STACKING;
    config.useMultiThreading = true;
    config.matchThreshold = 0.05;

    auto db_config = simple_integration::toDbConfig(config);

    // Test database creation and configuration recording
    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database should be enabled for integration test";

    int exp_id = db.recordConfiguration(db_config);
    EXPECT_GT(exp_id, 0) << "Configuration should be recorded with positive ID";

    // Test basic results recording
    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "RGBSIFT";
    results.dataset_name = "i_ajuntament";
    results.mean_average_precision = 0.87;
    results.processing_time_ms = 245.3;

    EXPECT_TRUE(db.recordExperiment(results)) << "Basic results should be recorded successfully";
}

TEST_F(DatabaseIntegrationSimpleTest, DetailedResultsRecording) {
    // Setup configuration
    MockExperimentConfig config;
    config.descriptorOptions.descriptorType = DESCRIPTOR_RGBSIFT;
    auto db_config = simple_integration::toDbConfig(config);

    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database should be enabled";

    int exp_id = db.recordConfiguration(db_config);
    ASSERT_GT(exp_id, 0) << "Configuration recording must succeed";

    // Test detailed results recording
    thesis_project::database::ExperimentResults detailed_results;
    detailed_results.experiment_id = exp_id;
    detailed_results.descriptor_type = "RGBSIFT";
    detailed_results.dataset_name = "i_ajuntament";
    detailed_results.mean_average_precision = 0.87;
    detailed_results.precision_at_1 = 0.92;
    detailed_results.precision_at_5 = 0.88;
    detailed_results.recall_at_1 = 0.85;
    detailed_results.recall_at_5 = 0.83;
    detailed_results.total_matches = 150;
    detailed_results.total_keypoints = 1000;
    detailed_results.processing_time_ms = 245.3;

    EXPECT_TRUE(db.recordExperiment(detailed_results)) << "Detailed results should be recorded successfully";

    // Verify results can be retrieved
    auto recent_results = db.getRecentResults(5);
    EXPECT_FALSE(recent_results.empty()) << "Should retrieve at least one result";
    
    if (!recent_results.empty()) {
        EXPECT_DOUBLE_EQ(recent_results[0].mean_average_precision, 0.87) << "Retrieved MAP should match";
        EXPECT_EQ(recent_results[0].descriptor_type, "RGBSIFT") << "Retrieved descriptor type should match";
        EXPECT_DOUBLE_EQ(recent_results[0].precision_at_1, 0.92) << "Retrieved precision@1 should match";
        EXPECT_EQ(recent_results[0].total_matches, 150) << "Retrieved match count should match";
    }
}

TEST_F(DatabaseIntegrationSimpleTest, StatisticsRetrieval) {
    // Setup and record some data
    MockExperimentConfig config;
    auto db_config = simple_integration::toDbConfig(config);

    thesis_project::database::DatabaseManager db(test_db_name, true);
    ASSERT_TRUE(db.isEnabled()) << "Database should be enabled";

    int exp_id = db.recordConfiguration(db_config);
    ASSERT_GT(exp_id, 0) << "Configuration recording must succeed";

    thesis_project::database::ExperimentResults results;
    results.experiment_id = exp_id;
    results.descriptor_type = "SIFT";
    results.dataset_name = "test_dataset";
    results.mean_average_precision = 0.75;
    results.processing_time_ms = 200.0;

    ASSERT_TRUE(db.recordExperiment(results)) << "Results recording must succeed";

    // Test statistics retrieval
    auto stats = db.getStatistics();
    EXPECT_FALSE(stats.empty()) << "Should retrieve some statistics";
    
    // Verify that statistics contain meaningful data
    bool has_meaningful_stats = false;
    for (const auto& [key, value] : stats) {
        EXPECT_FALSE(key.empty()) << "Statistic key should not be empty";
        if (key.find("experiment") != std::string::npos || 
            key.find("result") != std::string::npos ||
            key.find("count") != std::string::npos) {
            has_meaningful_stats = true;
        }
    }
    EXPECT_TRUE(has_meaningful_stats) << "Statistics should contain meaningful database metrics";
}

// Test all descriptor type conversions
TEST_F(DatabaseIntegrationSimpleTest, AllDescriptorTypeConversions) {
    std::vector<std::pair<DescriptorType, std::string>> type_mappings = {
        {DESCRIPTOR_SIFT, "SIFT"},
        {DESCRIPTOR_RGBSIFT, "RGBSIFT"},
        {DESCRIPTOR_HoNC, "HoNC"},
        {DESCRIPTOR_vSIFT, "vSIFT"}
    };

    for (const auto& [mock_type, expected_string] : type_mappings) {
        MockExperimentConfig config;
        config.descriptorOptions.descriptorType = mock_type;
        
        auto db_config = simple_integration::toDbConfig(config);
        EXPECT_EQ(db_config.descriptor_type, expected_string) 
            << "Descriptor type " << static_cast<int>(mock_type) << " should convert to " << expected_string;
    }
}

// Test all pooling strategy conversions
TEST_F(DatabaseIntegrationSimpleTest, AllPoolingStrategyConversions) {
    std::vector<std::pair<PoolingStrategy, std::string>> strategy_mappings = {
        {NONE, "NONE"},
        {STACKING, "STACKING"},
        {DOMAIN_SIZE_POOLING, "DOMAIN_SIZE_POOLING"}
    };

    for (const auto& [mock_strategy, expected_string] : strategy_mappings) {
        MockExperimentConfig config;
        config.descriptorOptions.poolingStrategy = mock_strategy;
        
        auto db_config = simple_integration::toDbConfig(config);
        EXPECT_EQ(db_config.pooling_strategy, expected_string) 
            << "Pooling strategy " << static_cast<int>(mock_strategy) << " should convert to " << expected_string;
    }
}