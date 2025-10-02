#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.hpp"
#include "include/thesis_project/types.hpp"

namespace {

using thesis_project::DescriptorParams;
using thesis_project::PoolingAggregation;
using thesis_project::PoolingStrategy;
using thesis_project::RootingStage;
using thesis_project::wrappers::DSPHoNCWrapperV2;
using thesis_project::wrappers::DSPHoWHWrapperV2;
using thesis_project::wrappers::DSPRGBSIFTWrapperV2;

DescriptorParams makeDefaultParams() {
    DescriptorParams params;
    params.pooling_aggregation = PoolingAggregation::AVERAGE;
    params.normalize_before_pooling = false;
    params.normalize_after_pooling = true;
    params.rooting_stage = RootingStage::R_AFTER_POOLING;
    return params;
}

template <typename WrapperT>
void expectDescriptorExtraction(const cv::Mat& image) {
    WrapperT wrapper;

    std::vector<cv::KeyPoint> keypoints;
    keypoints.emplace_back(cv::Point2f(16.0f, 16.0f), 12.0f, 0.0f, 1.0f, 0, -1);
    keypoints.emplace_back(cv::Point2f(24.0f, 24.0f), 14.0f, 45.0f, 0.8f, 1, -1);

    DescriptorParams params = makeDefaultParams();
    cv::Mat descriptors = wrapper.extract(image, keypoints, params);

    ASSERT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_GT(descriptors.cols, 0);
    EXPECT_EQ(descriptors.type(), CV_32F);
    EXPECT_EQ(wrapper.descriptorSize(), descriptors.cols);
}

} // namespace

TEST(DSPColorWrappers, ExtractsFromColorImage) {
    cv::Mat color_image(64, 64, CV_8UC3);
    cv::randu(color_image, 0, 255);

    expectDescriptorExtraction<DSPRGBSIFTWrapperV2>(color_image);
    expectDescriptorExtraction<DSPHoWHWrapperV2>(color_image);
    expectDescriptorExtraction<DSPHoNCWrapperV2>(color_image);
}

TEST(DSPColorWrappers, ExtractsFromGrayscaleImage) {
    cv::Mat gray_image(64, 64, CV_8UC1);
    cv::randu(gray_image, 0, 255);

    expectDescriptorExtraction<DSPRGBSIFTWrapperV2>(gray_image);
    expectDescriptorExtraction<DSPHoWHWrapperV2>(gray_image);
    expectDescriptorExtraction<DSPHoNCWrapperV2>(gray_image);
}
