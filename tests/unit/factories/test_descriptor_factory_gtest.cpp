#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "include/thesis_project/types.hpp"

namespace tf = thesis_project::factories;

TEST(DescriptorFactoryTest, CreateSIFT) {
    auto extractor = tf::DescriptorFactory::create(thesis_project::DescriptorType::SIFT);
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 128);
    EXPECT_EQ(extractor->descriptorType(), static_cast<int>(thesis_project::DescriptorType::SIFT));
}

TEST(DescriptorFactoryTest, CreateRGBSIFT) {
    auto extractor = tf::DescriptorFactory::create(thesis_project::DescriptorType::RGBSIFT);
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 384);
    EXPECT_EQ(extractor->descriptorType(), static_cast<int>(thesis_project::DescriptorType::RGBSIFT));
}

TEST(DescriptorFactoryTest, SupportsVariousTypes) {
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(thesis_project::DescriptorType::SIFT));
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(thesis_project::DescriptorType::RGBSIFT));
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(thesis_project::DescriptorType::HoNC));
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(thesis_project::DescriptorType::vSIFT));
}

TEST(DescriptorFactoryTest, SupportedTypesList) {
    auto types = tf::DescriptorFactory::getSupportedTypes();
    EXPECT_NE(std::find(types.begin(), types.end(), "SIFT"), types.end());
    EXPECT_NE(std::find(types.begin(), types.end(), "RGBSIFT"), types.end());
}

