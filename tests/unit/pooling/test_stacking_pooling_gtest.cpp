#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/config/experiment_config.hpp"

using thesis_project::pooling::PoolingFactory;

namespace {

cv::Mat makeColor(int w=220, int h=160) {
    cv::Mat img(h, w, CV_8UC3, cv::Scalar(10,20,30));
    cv::circle(img, {w/2, h/2}, std::min(w,h)/4, cv::Scalar(200,100,50), -1);
    return img;
}

std::vector<cv::KeyPoint> gridKeypoints(int w, int h, int step=24, int margin=24) {
    std::vector<cv::KeyPoint> kps;
    for (int y = margin; y < h - margin; y += step) {
        for (int x = margin; x < w - margin; x += step) {
            kps.emplace_back(static_cast<float>(x), static_cast<float>(y), 12.0f);
        }
    }
    return kps;
}

double rowL2(const cv::Mat& m, int r) {
    double s=0.0; const float* p = m.ptr<float>(r);
    for (int c=0;c<m.cols;++c) s += p[c]*p[c];
    return std::sqrt(s);
}

}

class StackingPoolingTest : public ::testing::Test {
protected:
    experiment_config cfg;
    cv::Mat image;
    std::vector<cv::KeyPoint> kps;

    void SetUp() override {
        cfg.descriptorOptions.poolingStrategy = STACKING;
        cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
        cfg.descriptorOptions.descriptorType2 = DESCRIPTOR_SIFT; // aligned dims
        cfg.descriptorOptions.descriptorColorSpace = D_BW;
        cfg.descriptorOptions.descriptorColorSpace2 = D_BW;
        cfg.descriptorOptions.normalizationStage = NO_NORMALIZATION;
        cfg.descriptorOptions.rootingStage = R_NONE;
        cfg.refreshDetectors();

        image = makeColor();
        // Provide BW; code will convert as needed
        cv::Mat gray; cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        image = gray;

        kps = gridKeypoints(image.cols, image.rows, 26, 30);
    }
};

TEST_F(StackingPoolingTest, DimensionalityAndType) {
    auto stacking = PoolingFactory::createStrategy(STACKING);
    cv::Mat stacked = stacking->computeDescriptors(image, kps, cfg.detector, cfg);
    ASSERT_FALSE(stacked.empty());
    EXPECT_EQ(stacked.rows, static_cast<int>(kps.size()));
    EXPECT_EQ(stacked.type(), CV_32F);
    // SIFT (128) + SIFT (128) = 256 cols
    EXPECT_EQ(stacked.cols, 256);
}

TEST_F(StackingPoolingTest, AfterPoolingL2RowWiseNormalization) {
    cfg.descriptorOptions.normalizationStage = AFTER_POOLING;
    cfg.descriptorOptions.normType = cv::NORM_L2;
    auto stacking = PoolingFactory::createStrategy(STACKING);
    cv::Mat stacked = stacking->computeDescriptors(image, kps, cfg.detector, cfg);
    ASSERT_FALSE(stacked.empty());
    // Idempotence: re-normalizing rows by L2 should not change values
    cv::Mat check = stacked.clone();
    for (int r=0;r<check.rows;++r) {
        cv::Mat row = check.row(r);
        cv::normalize(row, row, 1.0, 0.0, cv::NORM_L2);
    }
    ASSERT_EQ(stacked.size, check.size);
    // Compare a few rows for efficiency
    for (int r=0;r<std::min(5, stacked.rows); ++r) {
        EXPECT_NEAR(rowL2(stacked, r), rowL2(check, r), 1e-6);
    }
}

TEST_F(StackingPoolingTest, MissingSecondaryDetectorReturnsEmpty) {
    // Intentionally do not provision detector2 for stacking
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
    cfg.descriptorOptions.descriptorType2 = NO_DESCRIPTOR;
    cfg.refreshDetectors();
    cfg.detector2.release(); // ensure missing secondary detector
    auto stacking = PoolingFactory::createStrategy(STACKING);
    cv::Mat stacked = stacking->computeDescriptors(image, kps, cfg.detector, cfg);
    EXPECT_TRUE(stacked.empty());
}

TEST_F(StackingPoolingTest, SIFTPlusRGBSIFTDimsAndColorHandling) {
    // Prepare color image for RGBSIFT
    image = makeColor();
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
    cfg.descriptorOptions.descriptorType2 = DESCRIPTOR_RGBSIFT;
    cfg.descriptorOptions.descriptorColorSpace = D_BW;
    cfg.descriptorOptions.descriptorColorSpace2 = D_COLOR;
    cfg.refreshDetectors();

    auto stacking = PoolingFactory::createStrategy(STACKING);
    cv::Mat stacked = stacking->computeDescriptors(image, kps, cfg.detector, cfg);
    ASSERT_FALSE(stacked.empty());
    EXPECT_EQ(stacked.rows, static_cast<int>(kps.size()));
    EXPECT_EQ(stacked.type(), CV_32F);
    // 128 (SIFT) + 384 (RGBSIFT) = 512
    EXPECT_EQ(stacked.cols, 512);
}

TEST_F(StackingPoolingTest, RootSIFTBeforePoolingNonNegative) {
    cfg.descriptorOptions.normalizationStage = NO_NORMALIZATION;
    cfg.descriptorOptions.rootingStage = R_BEFORE_POOLING;
    auto stacking = PoolingFactory::createStrategy(STACKING);
    cv::Mat stacked = stacking->computeDescriptors(image, kps, cfg.detector, cfg);
    ASSERT_FALSE(stacked.empty());
    for (int r=0;r<stacked.rows;++r) {
        const float* p = stacked.ptr<float>(r);
        for (int c=0;c<stacked.cols;++c) {
            ASSERT_GE(p[c], 0.0f);
        }
    }
}
