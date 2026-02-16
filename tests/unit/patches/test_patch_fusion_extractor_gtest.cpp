#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <cmath>

#include "src/core/patches/PatchFusionExtractor.hpp"
#include "src/core/patches/PatchDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"

using namespace thesis_project;
using namespace thesis_project::patches;

// Mock extractor that returns predictable descriptors for testing fusion logic
class MockPatchExtractor : public IPatchDescriptorExtractor {
public:
    MockPatchExtractor(int dim, float base_value, const std::string& name)
        : dim_(dim), base_value_(base_value), name_(name) {}

    cv::Mat extractFromPatches(const std::vector<cv::Mat>& patches,
                               const DescriptorParams& /*params*/) override {
        cv::Mat descriptors(static_cast<int>(patches.size()), dim_, CV_32F);
        descriptors.setTo(base_value_);
        return descriptors;
    }

    int descriptorSize() const override { return dim_; }
    int descriptorType() const override { return CV_32F; }
    std::string name() const override { return name_; }
    bool requiresResize() const override { return false; }
    int expectedPatchSize() const override { return 65; }

    std::unique_ptr<IPatchDescriptorExtractor> clone() const override {
        return std::make_unique<MockPatchExtractor>(dim_, base_value_, name_);
    }

private:
    int dim_;
    float base_value_;
    std::string name_;
};

// Test fixture for PatchFusionExtractor tests
class PatchFusionExtractorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create some dummy patches
        patches_.clear();
        for (int i = 0; i < 3; ++i) {
            cv::Mat patch(65, 65, CV_8UC1);
            cv::randu(patch, 0, 255);
            patches_.push_back(patch);
        }
    }

    std::vector<cv::Mat> patches_;
};

// ============================================================================
// L2 Normalization Tests (Critical - verifies the fix we made)
// ============================================================================

TEST_F(PatchFusionExtractorTest, NormalizesComponentsBeforeFusion) {
    // Create two extractors with very different magnitudes
    // This simulates SIFT (~0-512 range) vs HardNet (~0-1 range)
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 100.0f, "high_mag"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 0.1f, "low_mag"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    // After proper normalization, both components should contribute equally
    // The result should be L2 normalized (unit norm)
    for (int i = 0; i < result.rows; ++i) {
        cv::Mat row = result.row(i);
        double norm = cv::norm(row, cv::NORM_L2);
        EXPECT_NEAR(norm, 1.0, 1e-5) << "Fused descriptor should be unit normalized";
    }
}

TEST_F(PatchFusionExtractorTest, OutputIsUnitNormalized) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 2.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    // Every row should have L2 norm of 1.0
    for (int i = 0; i < result.rows; ++i) {
        cv::Mat row = result.row(i);
        double norm = cv::norm(row, cv::NORM_L2);
        EXPECT_NEAR(norm, 1.0, 1e-5) << "Row " << i << " should be unit normalized";
    }
}

TEST_F(PatchFusionExtractorTest, EqualContributionAfterNormalization) {
    // Create two extractors with different magnitudes but same direction
    // After normalization, they should contribute equally to average
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 500.0f, "sift_like"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "cnn_like"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    // Since both components have uniform values that become unit vectors,
    // and we average them, the result should still be a uniform unit vector
    // All elements should be equal (within floating point tolerance)
    for (int i = 0; i < result.rows; ++i) {
        float first_val = result.at<float>(i, 0);
        for (int j = 1; j < result.cols; ++j) {
            EXPECT_NEAR(result.at<float>(i, j), first_val, 1e-5)
                << "All elements should be equal after averaging uniform unit vectors";
        }
    }
}

// ============================================================================
// Fusion Method Tests
// ============================================================================

TEST_F(PatchFusionExtractorTest, AverageFusionCorrectDimensions) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    EXPECT_EQ(fusion.descriptorSize(), 128);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    EXPECT_EQ(result.rows, static_cast<int>(patches_.size()));
    EXPECT_EQ(result.cols, 128);
}

TEST_F(PatchFusionExtractorTest, ConcatenateFusionCorrectDimensions) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(64, 1.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::CONCATENATE);

    EXPECT_EQ(fusion.descriptorSize(), 192);  // 128 + 64

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    EXPECT_EQ(result.rows, static_cast<int>(patches_.size()));
    EXPECT_EQ(result.cols, 192);
}

TEST_F(PatchFusionExtractorTest, MaxFusionCorrectDimensions) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 2.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::MAX);

    EXPECT_EQ(fusion.descriptorSize(), 128);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    EXPECT_EQ(result.cols, 128);
}

TEST_F(PatchFusionExtractorTest, MinFusionCorrectDimensions) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 0.5f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::MIN);

    EXPECT_EQ(fusion.descriptorSize(), 128);
}

TEST_F(PatchFusionExtractorTest, WeightedAverageFusion) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp2"));

    std::vector<float> weights = {0.7f, 0.3f};
    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::WEIGHTED_AVG, weights);

    DescriptorParams params;
    cv::Mat result = fusion.extractFromPatches(patches_, params);

    // Result should still be unit normalized
    for (int i = 0; i < result.rows; ++i) {
        cv::Mat row = result.row(i);
        double norm = cv::norm(row, cv::NORM_L2);
        EXPECT_NEAR(norm, 1.0, 1e-5);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(PatchFusionExtractorTest, HandlesEmptyPatches) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    DescriptorParams params;
    std::vector<cv::Mat> empty_patches;
    cv::Mat result = fusion.extractFromPatches(empty_patches, params);

    EXPECT_TRUE(result.empty());
}

TEST_F(PatchFusionExtractorTest, ThrowsOnDimensionMismatchForAverage) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(256, 1.0f, "comp2"));

    // Average requires same dimensions
    EXPECT_THROW(
        PatchFusionExtractor(std::move(components), PatchFusionMethod::AVERAGE),
        std::invalid_argument
    );
}

TEST_F(PatchFusionExtractorTest, ThrowsOnEmptyComponents) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> empty_components;

    EXPECT_THROW(
        PatchFusionExtractor(std::move(empty_components), PatchFusionMethod::AVERAGE),
        std::invalid_argument
    );
}

// ============================================================================
// Name Generation Tests
// ============================================================================

TEST_F(PatchFusionExtractorTest, GeneratesCorrectName) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "sift"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "hardnet"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);

    std::string name = fusion.name();
    EXPECT_TRUE(name.find("sift") != std::string::npos);
    EXPECT_TRUE(name.find("hardnet") != std::string::npos);
    EXPECT_TRUE(name.find("average") != std::string::npos);
}

TEST_F(PatchFusionExtractorTest, CustomNameOverridesGenerated) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE, {}, "my_custom_name");

    EXPECT_EQ(fusion.name(), "my_custom_name");
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST_F(PatchFusionExtractorTest, ClonePreservesConfiguration) {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp1"));
    components.push_back(std::make_unique<MockPatchExtractor>(128, 1.0f, "comp2"));

    PatchFusionExtractor fusion(std::move(components), PatchFusionMethod::AVERAGE);
    auto cloned = fusion.clone();

    EXPECT_EQ(cloned->descriptorSize(), fusion.descriptorSize());
    EXPECT_EQ(cloned->name(), fusion.name());
}
