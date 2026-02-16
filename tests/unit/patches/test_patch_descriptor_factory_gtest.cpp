#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "src/core/patches/PatchDescriptorFactory.hpp"
#include "include/thesis_project/types.hpp"

using namespace thesis_project;
using namespace thesis_project::patches;

// ============================================================================
// Type Creation Tests
// ============================================================================

TEST(PatchDescriptorFactoryTest, CreatesSIFT) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::SIFT);
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 128);
    // Name comes from the underlying OpenCV extractor
    EXPECT_EQ(extractor->name(), "SIFT");
}

TEST(PatchDescriptorFactoryTest, CreatesRGBSIFT) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::RGBSIFT);
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 384);
}

TEST(PatchDescriptorFactoryTest, CreatesRGBSIFTChannelAvg) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::RGBSIFT_CHANNEL_AVG);
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 128);
}

TEST(PatchDescriptorFactoryTest, CreatesHoNC) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::HoNC);
    ASSERT_NE(extractor, nullptr);
}

TEST(PatchDescriptorFactoryTest, CreatesSURF) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::SURF);
    ASSERT_NE(extractor, nullptr);
}

TEST(PatchDescriptorFactoryTest, CreatesDSPSIFT) {
    auto extractor = PatchDescriptorFactory::create(DescriptorType::DSPSIFT_V2);
    ASSERT_NE(extractor, nullptr);
}

// ============================================================================
// String to Type Conversion Tests
// ============================================================================

TEST(PatchDescriptorFactoryTest, CreatesByStringName) {
    auto extractor = PatchDescriptorFactory::create("sift");
    ASSERT_NE(extractor, nullptr);
    EXPECT_EQ(extractor->descriptorSize(), 128);
}

TEST(PatchDescriptorFactoryTest, StringConversionCaseInsensitive) {
    auto extractor1 = PatchDescriptorFactory::create("SIFT");
    auto extractor2 = PatchDescriptorFactory::create("sift");
    auto extractor3 = PatchDescriptorFactory::create("Sift");

    ASSERT_NE(extractor1, nullptr);
    ASSERT_NE(extractor2, nullptr);
    ASSERT_NE(extractor3, nullptr);

    EXPECT_EQ(extractor1->descriptorSize(), extractor2->descriptorSize());
    EXPECT_EQ(extractor2->descriptorSize(), extractor3->descriptorSize());
}

TEST(PatchDescriptorFactoryTest, StringToTypeConversionsViaCreate) {
    // Test string to type conversion by verifying create() works with various strings
    // (stringToType is private, so we test it indirectly through create())
    EXPECT_NE(PatchDescriptorFactory::create("sift"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("rgbsift"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("rgbsift_channel_avg"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("honc"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("dspsift"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("dspsift_v2"), nullptr);
    EXPECT_NE(PatchDescriptorFactory::create("surf"), nullptr);
}

TEST(PatchDescriptorFactoryTest, ThrowsOnUnknownType) {
    EXPECT_THROW(
        PatchDescriptorFactory::create("unknown_descriptor"),
        std::invalid_argument
    );
}

// ============================================================================
// Supported Types Tests
// ============================================================================

TEST(PatchDescriptorFactoryTest, IsSupportedReturnsCorrectly) {
    EXPECT_TRUE(PatchDescriptorFactory::isSupported(DescriptorType::SIFT));
    EXPECT_TRUE(PatchDescriptorFactory::isSupported(DescriptorType::RGBSIFT));
    EXPECT_TRUE(PatchDescriptorFactory::isSupported(DescriptorType::HoNC));
    EXPECT_TRUE(PatchDescriptorFactory::isSupported(DescriptorType::SURF));
    EXPECT_TRUE(PatchDescriptorFactory::isSupported(DescriptorType::DSPSIFT_V2));

    // CNN types require LibTorch build
    // These may or may not be supported depending on build configuration
}

TEST(PatchDescriptorFactoryTest, SupportedTypesListNotEmpty) {
    auto types = PatchDescriptorFactory::supportedTypes();
    EXPECT_FALSE(types.empty());
    EXPECT_GE(types.size(), 5);  // At least SIFT, RGBSIFT, HoNC, SURF, DSPSIFT
}

TEST(PatchDescriptorFactoryTest, SupportedTypesContainsBasicTypes) {
    auto types = PatchDescriptorFactory::supportedTypes();

    auto contains = [&types](const std::string& name) {
        return std::find(types.begin(), types.end(), name) != types.end();
    };

    EXPECT_TRUE(contains("sift"));
    EXPECT_TRUE(contains("rgbsift"));
    EXPECT_TRUE(contains("honc"));
    EXPECT_TRUE(contains("surf"));
}

// ============================================================================
// Fusion Creation Tests
// ============================================================================

TEST(PatchDescriptorFactoryTest, CreatesFusionFromTypes) {
    std::vector<DescriptorType> components = {
        DescriptorType::SIFT,
        DescriptorType::SIFT
    };

    auto fusion = PatchDescriptorFactory::createFusion(
        components,
        PatchFusionMethod::AVERAGE
    );

    ASSERT_NE(fusion, nullptr);
    EXPECT_EQ(fusion->descriptorSize(), 128);  // Average preserves dimension
}

TEST(PatchDescriptorFactoryTest, CreatesFusionFromStrings) {
    std::vector<std::string> components = {"sift", "sift"};

    auto fusion = PatchDescriptorFactory::createFusion(
        components,
        "average"
    );

    ASSERT_NE(fusion, nullptr);
    EXPECT_EQ(fusion->descriptorSize(), 128);
}

TEST(PatchDescriptorFactoryTest, CreatesConcatenateFusion) {
    std::vector<DescriptorType> components = {
        DescriptorType::SIFT,
        DescriptorType::SIFT
    };

    auto fusion = PatchDescriptorFactory::createFusion(
        components,
        PatchFusionMethod::CONCATENATE
    );

    ASSERT_NE(fusion, nullptr);
    EXPECT_EQ(fusion->descriptorSize(), 256);  // 128 + 128
}

TEST(PatchDescriptorFactoryTest, FusionWithWeights) {
    std::vector<DescriptorType> components = {
        DescriptorType::SIFT,
        DescriptorType::SIFT
    };
    std::vector<float> weights = {0.6f, 0.4f};

    auto fusion = PatchDescriptorFactory::createFusion(
        components,
        PatchFusionMethod::WEIGHTED_AVG,
        weights
    );

    ASSERT_NE(fusion, nullptr);
}

TEST(PatchDescriptorFactoryTest, FusionWithCustomName) {
    std::vector<DescriptorType> components = {
        DescriptorType::SIFT,
        DescriptorType::SIFT
    };

    auto fusion = PatchDescriptorFactory::createFusion(
        components,
        PatchFusionMethod::AVERAGE,
        {},
        "my_fusion"
    );

    ASSERT_NE(fusion, nullptr);
    EXPECT_EQ(fusion->name(), "my_fusion");
}

// ============================================================================
// Unsupported Type Tests
// ============================================================================

TEST(PatchDescriptorFactoryTest, ThrowsOnUnsupportedStringType) {
    // Test with a type string that's not supported in patch pipeline
    EXPECT_THROW(
        PatchDescriptorFactory::create("nonexistent_descriptor_type"),
        std::invalid_argument
    );
}
