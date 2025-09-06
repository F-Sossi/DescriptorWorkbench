#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "src/core/config/experiment_config.hpp"

namespace tf = thesis_project::factories;

class DescriptorFactoryTest : public ::testing::Test {
protected:
    experiment_config cfg;
};

TEST_F(DescriptorFactoryTest, CreateSIFT) {
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
    auto ext = tf::DescriptorFactory::create(cfg);
    ASSERT_NE(ext, nullptr);
    EXPECT_FALSE(ext->name().empty());
    EXPECT_EQ(ext->descriptorSize(), 128);
    EXPECT_EQ(ext->descriptorType(), DESCRIPTOR_SIFT);
}

TEST_F(DescriptorFactoryTest, TryCreateRGBSIFT) {
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_RGBSIFT;
    auto ext = tf::DescriptorFactory::tryCreate(cfg);
    ASSERT_NE(ext, nullptr);
    EXPECT_EQ(ext->descriptorSize(), 384);
    EXPECT_EQ(ext->descriptorType(), DESCRIPTOR_RGBSIFT);
}

TEST_F(DescriptorFactoryTest, SupportHoNCAndVSIFT) {
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_HoNC;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
    auto honc = tf::DescriptorFactory::tryCreate(cfg);
    ASSERT_NE(honc, nullptr);
    EXPECT_EQ(honc->descriptorType(), DESCRIPTOR_HoNC);

    cfg.descriptorOptions.descriptorType = DESCRIPTOR_vSIFT;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
    auto vsift = tf::DescriptorFactory::tryCreate(cfg);
    ASSERT_NE(vsift, nullptr);
    EXPECT_EQ(vsift->descriptorType(), DESCRIPTOR_vSIFT);
}

TEST_F(DescriptorFactoryTest, SupportedTypesList) {
    auto types = tf::DescriptorFactory::getSupportedTypes();
    bool has_sift = false, has_rgbsift = false, has_honc = false, has_vsift = false, has_vgg = false;
    for (const auto& t : types) {
        if (t == "SIFT") has_sift = true;
        if (t == "RGBSIFT") has_rgbsift = true;
        if (t == "HoNC") has_honc = true;
        if (t == "VSIFT") has_vsift = true;
        if (t == "VGG") has_vgg = true;
    }
    EXPECT_TRUE(has_sift);
    EXPECT_TRUE(has_rgbsift);
    EXPECT_TRUE(has_honc);
    EXPECT_TRUE(has_vsift);
    // VGG is optional depending on OpenCV build; no hard assertion
}

TEST_F(DescriptorFactoryTest, IsSupportedFlags) {
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_SIFT;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_RGBSIFT;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_HoNC;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
    cfg.descriptorOptions.descriptorType = DESCRIPTOR_vSIFT;
    EXPECT_TRUE(tf::DescriptorFactory::isSupported(cfg));
}
