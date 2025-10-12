#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "src/core/descriptor/extractors/wrappers/DSPSIFTWrapperV2.hpp"
#include "include/thesis_project/types.hpp"

using namespace thesis_project;

// Test fixture for DSP SIFT Wrapper V2 (pyramid reuse tests)
class DSPSIFTWrapperV2Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple grayscale image with structure
        image_ = cv::Mat(128, 128, CV_8UC1);
        cv::randu(image_, 0, 255);

        // Create keypoints at different scales
        keypoints_.clear();
        keypoints_.emplace_back(cv::Point2f(32.0f, 32.0f), 8.0f, 0.0f, 1.0f, 0, -1);
        keypoints_.emplace_back(cv::Point2f(64.0f, 64.0f), 16.0f, 45.0f, 0.8f, 1, -1);
        keypoints_.emplace_back(cv::Point2f(96.0f, 96.0f), 32.0f, 90.0f, 0.6f, 2, -1);
    }

    DescriptorParams makeParams(PoolingAggregation agg = PoolingAggregation::AVERAGE) {
        DescriptorParams params;
        params.pooling_aggregation = agg;
        params.scales = {0.85f, 1.0f, 1.30f};  // DSP default scales
        params.normalize_before_pooling = false;
        params.normalize_after_pooling = true;
        params.rooting_stage = RootingStage::R_AFTER_POOLING;
        params.norm_type = cv::NORM_L2;
        return params;
    }

    cv::Mat image_;
    std::vector<cv::KeyPoint> keypoints_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(DSPSIFTWrapperV2Test, ExtractsDescriptorsFromGrayscaleImage) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty()) << "Descriptors should not be empty";
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints_.size())) << "Should have one descriptor per keypoint";
    EXPECT_EQ(descriptors.cols, 128) << "SIFT descriptors should be 128-dimensional";
    EXPECT_EQ(descriptors.type(), CV_32F) << "Descriptors should be float type";
}

TEST_F(DSPSIFTWrapperV2Test, ExtractsDescriptorsFromColorImage) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();

    cv::Mat color_image;
    cv::cvtColor(image_, color_image, cv::COLOR_GRAY2BGR);

    cv::Mat descriptors = wrapper.extract(color_image, keypoints_, params);

    ASSERT_FALSE(descriptors.empty()) << "Descriptors should not be empty for color images";
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints_.size()));
    EXPECT_EQ(descriptors.cols, 128);
}

TEST_F(DSPSIFTWrapperV2Test, HandlesEmptyKeypoints) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();
    std::vector<cv::KeyPoint> empty_keypoints;

    cv::Mat descriptors = wrapper.extract(image_, empty_keypoints, params);

    EXPECT_TRUE(descriptors.empty()) << "Should return empty descriptors for empty keypoints";
}

TEST_F(DSPSIFTWrapperV2Test, DescriptorSizeMatchesOutput) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    EXPECT_EQ(wrapper.descriptorSize(), descriptors.cols) << "descriptorSize() should match actual descriptor dimension";
}

TEST_F(DSPSIFTWrapperV2Test, DescriptorTypeIsFloat) {
    DSPSIFTWrapperV2 wrapper;

    EXPECT_EQ(wrapper.descriptorType(), CV_32F) << "Descriptor type should be CV_32F";
}

TEST_F(DSPSIFTWrapperV2Test, NameIsCorrect) {
    DSPSIFTWrapperV2 wrapper;

    EXPECT_EQ(wrapper.name(), "DSPSIFTWrapperV2") << "Wrapper name should be correct";
}

// ============================================================================
// Aggregation Method Tests (Core DSP Functionality)
// ============================================================================

TEST_F(DSPSIFTWrapperV2Test, AverageAggregation_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams(PoolingAggregation::AVERAGE);

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.cols, 128) << "AVERAGE aggregation should maintain 128D";

    // Check that descriptors contain valid values (not all zeros)
    bool has_nonzero = false;
    for (int r = 0; r < descriptors.rows; ++r) {
        double norm = cv::norm(descriptors.row(r));
        if (norm > 0.01) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Descriptors should contain non-zero values";
}

TEST_F(DSPSIFTWrapperV2Test, MaxAggregation_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams(PoolingAggregation::MAX);

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.cols, 128) << "MAX aggregation should maintain 128D";

    // MAX pooling should produce non-negative values
    double min_val;
    cv::minMaxLoc(descriptors, &min_val, nullptr);
    EXPECT_GE(min_val, 0.0) << "MAX pooling should produce non-negative values";
}

TEST_F(DSPSIFTWrapperV2Test, MinAggregation_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams(PoolingAggregation::MIN);

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.cols, 128) << "MIN aggregation should maintain 128D";
}

TEST_F(DSPSIFTWrapperV2Test, ConcatenateAggregation_IncreasesDimensionality) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams(PoolingAggregation::CONCATENATE);

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    // CONCATENATE with 3 scales should produce 128*3 = 384 dimensions
    EXPECT_EQ(descriptors.cols, 128 * 3) << "CONCATENATE should multiply dimensionality by number of scales";
}

TEST_F(DSPSIFTWrapperV2Test, WeightedAverageAggregation_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams(PoolingAggregation::WEIGHTED_AVG);
    params.scale_weighting = ScaleWeighting::UNIFORM;

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.cols, 128) << "WEIGHTED_AVG aggregation should maintain 128D";
}

// ============================================================================
// Multi-Scale Pyramid Tests (Pyramid Reuse Validation)
// ============================================================================

TEST_F(DSPSIFTWrapperV2Test, MultipleScales_ProduceDifferentResults) {
    DSPSIFTWrapperV2 wrapper;

    // Test with single scale (no pooling)
    DescriptorParams single_scale_params = makeParams();
    single_scale_params.scales = {1.0f};
    cv::Mat single_scale_desc = wrapper.extract(image_, keypoints_, single_scale_params);

    // Test with multiple scales (DSP pooling)
    DescriptorParams multi_scale_params = makeParams();
    multi_scale_params.scales = {0.85f, 1.0f, 1.30f};
    cv::Mat multi_scale_desc = wrapper.extract(image_, keypoints_, multi_scale_params);

    ASSERT_EQ(single_scale_desc.rows, multi_scale_desc.rows);
    ASSERT_EQ(single_scale_desc.cols, multi_scale_desc.cols);

    // Descriptors should be different due to multi-scale pooling
    bool descriptors_differ = false;
    for (int r = 0; r < single_scale_desc.rows; ++r) {
        double distance = cv::norm(single_scale_desc.row(r), multi_scale_desc.row(r), cv::NORM_L2);
        if (distance > 0.01) {  // Allow small numerical differences
            descriptors_differ = true;
            break;
        }
    }
    EXPECT_TRUE(descriptors_differ) << "Multi-scale descriptors should differ from single-scale";
}

TEST_F(DSPSIFTWrapperV2Test, DifferentAggregations_ProduceDifferentResults) {
    DSPSIFTWrapperV2 wrapper;

    DescriptorParams avg_params = makeParams(PoolingAggregation::AVERAGE);
    cv::Mat avg_desc = wrapper.extract(image_, keypoints_, avg_params);

    DescriptorParams max_params = makeParams(PoolingAggregation::MAX);
    cv::Mat max_desc = wrapper.extract(image_, keypoints_, max_params);

    ASSERT_EQ(avg_desc.rows, max_desc.rows);
    ASSERT_EQ(avg_desc.cols, max_desc.cols);

    // Different aggregation methods should produce different results
    bool descriptors_differ = false;
    for (int r = 0; r < avg_desc.rows; ++r) {
        double distance = cv::norm(avg_desc.row(r), max_desc.row(r), cv::NORM_L2);
        if (distance > 0.01) {
            descriptors_differ = true;
            break;
        }
    }
    EXPECT_TRUE(descriptors_differ) << "AVERAGE and MAX aggregation should produce different results";
}

// ============================================================================
// Normalization and Rooting Tests
// ============================================================================

TEST_F(DSPSIFTWrapperV2Test, NormalizationAfterPooling_ProducesUnitVectors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();
    params.normalize_after_pooling = true;
    params.norm_type = cv::NORM_L2;

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());

    // Check that descriptors are approximately unit length (L2 normalized)
    for (int r = 0; r < descriptors.rows; ++r) {
        double norm = cv::norm(descriptors.row(r), cv::NORM_L2);
        EXPECT_NEAR(norm, 1.0, 0.01) << "Descriptor " << r << " should be L2 normalized";
    }
}

TEST_F(DSPSIFTWrapperV2Test, RootingAfterPooling_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();
    params.rooting_stage = RootingStage::R_AFTER_POOLING;
    params.normalize_after_pooling = true;

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());

    // All values should be non-negative (square root applied)
    double min_val;
    cv::minMaxLoc(descriptors, &min_val, nullptr);
    EXPECT_GE(min_val, -0.001) << "Rooting should produce non-negative values";
}

TEST_F(DSPSIFTWrapperV2Test, RootingBeforePooling_ProducesValidDescriptors) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();
    params.rooting_stage = RootingStage::R_BEFORE_POOLING;

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints_.size()));
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

TEST_F(DSPSIFTWrapperV2Test, ProducesNonEmptyDescriptorsForAllKeypoints) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();

    cv::Mat descriptors = wrapper.extract(image_, keypoints_, params);

    ASSERT_EQ(descriptors.rows, static_cast<int>(keypoints_.size()));

    // Check that all descriptors are non-empty
    for (int r = 0; r < descriptors.rows; ++r) {
        double norm = cv::norm(descriptors.row(r));
        EXPECT_GT(norm, 0.0) << "Descriptor " << r << " should not be all zeros";
    }
}

TEST_F(DSPSIFTWrapperV2Test, ConsistentResultsAcrossMultipleExtractions) {
    DSPSIFTWrapperV2 wrapper;
    DescriptorParams params = makeParams();

    cv::Mat desc1 = wrapper.extract(image_, keypoints_, params);
    cv::Mat desc2 = wrapper.extract(image_, keypoints_, params);

    ASSERT_EQ(desc1.rows, desc2.rows);
    ASSERT_EQ(desc1.cols, desc2.cols);

    // Results should be identical for same input
    double max_diff = 0.0;
    for (int r = 0; r < desc1.rows; ++r) {
        double distance = cv::norm(desc1.row(r), desc2.row(r), cv::NORM_L2);
        max_diff = std::max(max_diff, distance);
    }
    EXPECT_LT(max_diff, 0.001) << "Repeated extractions should produce identical results";
}
