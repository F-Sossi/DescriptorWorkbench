#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

#include "src/core/patches/PatchTraditionalExtractor.hpp"
#include "src/core/patches/PatchLoader.hpp"
#include "include/thesis_project/types.hpp"

using namespace thesis_project;
using namespace thesis_project::patches;

// Test fixture for PatchTraditionalExtractor tests
class PatchTraditionalExtractorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create synthetic 65x65 patches with some structure
        patches_.clear();
        for (int i = 0; i < 5; ++i) {
            cv::Mat patch(PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
            // Create a gradient pattern for visual structure
            for (int y = 0; y < patch.rows; ++y) {
                for (int x = 0; x < patch.cols; ++x) {
                    patch.at<uchar>(y, x) = static_cast<uchar>((x + y + i * 20) % 256);
                }
            }
            patches_.push_back(patch);
        }

        // Create color patches
        color_patches_.clear();
        for (int i = 0; i < 3; ++i) {
            cv::Mat patch(PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC3);
            cv::randu(patch, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            color_patches_.push_back(patch);
        }
    }

    std::vector<cv::Mat> patches_;
    std::vector<cv::Mat> color_patches_;
};

// ============================================================================
// SIFT Extractor Tests
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, SIFTExtractsCorrectDimensions) {
    auto extractor = createPatchSIFT();

    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(patches_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(patches_.size()));
    EXPECT_EQ(descriptors.cols, 128);  // SIFT is 128-dimensional
    EXPECT_EQ(descriptors.type(), CV_32F);
}

TEST_F(PatchTraditionalExtractorTest, SIFTReturnsEmptyForEmptyInput) {
    auto extractor = createPatchSIFT();

    DescriptorParams params;
    std::vector<cv::Mat> empty_patches;
    cv::Mat descriptors = extractor->extractFromPatches(empty_patches, params);

    EXPECT_TRUE(descriptors.empty());
}

TEST_F(PatchTraditionalExtractorTest, SIFTDescriptorsHaveReasonableNorm) {
    auto extractor = createPatchSIFT();

    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(patches_, params);

    // OpenCV SIFT descriptors are NOT unit normalized - values are typically 0-512
    // Check that each descriptor has non-zero norm (valid extraction)
    for (int i = 0; i < descriptors.rows; ++i) {
        cv::Mat row = descriptors.row(i);
        double norm = cv::norm(row, cv::NORM_L2);
        EXPECT_GT(norm, 0.0) << "Descriptor " << i << " should have non-zero norm";
        // SIFT descriptors typically have L2 norm around 512 for non-trivial patches
        EXPECT_LT(norm, 600.0) << "Descriptor " << i << " has unusually high norm: " << norm;
    }
}

TEST_F(PatchTraditionalExtractorTest, SIFTDescriptorSize) {
    auto extractor = createPatchSIFT();
    EXPECT_EQ(extractor->descriptorSize(), 128);
}

TEST_F(PatchTraditionalExtractorTest, SIFTName) {
    auto extractor = createPatchSIFT();
    // Name comes from the underlying OpenCV extractor
    EXPECT_EQ(extractor->name(), "SIFT");
}

// ============================================================================
// RGBSIFT Extractor Tests
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, RGBSIFTExtractsCorrectDimensions) {
    auto extractor = createPatchRGBSIFT();

    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(color_patches_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(color_patches_.size()));
    EXPECT_EQ(descriptors.cols, 384);  // RGBSIFT is 3x128 = 384-dimensional
    EXPECT_EQ(descriptors.type(), CV_32F);
}

TEST_F(PatchTraditionalExtractorTest, RGBSIFTDescriptorSize) {
    auto extractor = createPatchRGBSIFT();
    EXPECT_EQ(extractor->descriptorSize(), 384);
}

// ============================================================================
// RGBSIFT Channel Average Tests
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, RGBSIFTChannelAvgExtractsCorrectDimensions) {
    auto extractor = createPatchRGBSIFTChannelAvg();

    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(color_patches_, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(color_patches_.size()));
    EXPECT_EQ(descriptors.cols, 128);  // Channel average reduces back to 128D
    EXPECT_EQ(descriptors.type(), CV_32F);
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, CloneProducesSameResults) {
    auto extractor = createPatchSIFT();
    auto cloned = extractor->clone();

    DescriptorParams params;
    cv::Mat desc1 = extractor->extractFromPatches(patches_, params);
    cv::Mat desc2 = cloned->extractFromPatches(patches_, params);

    ASSERT_EQ(desc1.size(), desc2.size());

    // Descriptors should be identical (or very close due to floating point)
    double diff = cv::norm(desc1 - desc2, cv::NORM_L2);
    EXPECT_LT(diff, 1e-5) << "Cloned extractor should produce identical results";
}

// ============================================================================
// Empty Patch Handling Tests
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, HandlesAllValidPatches) {
    auto extractor = createPatchSIFT();

    // All valid patches should produce descriptors
    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(patches_, params);

    EXPECT_EQ(descriptors.rows, static_cast<int>(patches_.size()));
    EXPECT_FALSE(descriptors.empty());
}

// ============================================================================
// Grayscale vs Color Handling
// ============================================================================

TEST_F(PatchTraditionalExtractorTest, SIFTConvertsColorToGray) {
    auto extractor = createPatchSIFT();

    DescriptorParams params;
    cv::Mat descriptors = extractor->extractFromPatches(color_patches_, params);

    // SIFT should handle color input by converting to grayscale
    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(color_patches_.size()));
    EXPECT_EQ(descriptors.cols, 128);
}
