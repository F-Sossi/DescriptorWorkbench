#include <gtest/gtest.h>

// Include constants directly - same as original test
namespace thesis_project {
    namespace constants {
        constexpr int SIFT_DESCR_WIDTH = 4;
        constexpr int SIFT_DESCR_HIST_BINS = 8;
        constexpr int SIFT_DESCRIPTOR_SIZE = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
        constexpr int RGB_SIFT_DESCRIPTOR_SIZE = 3 * SIFT_DESCRIPTOR_SIZE;
    }
}

class ConstantsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed for constants tests
    }
};

TEST_F(ConstantsTest, SIFTConstants) {
    using namespace thesis_project::constants;
    
    EXPECT_EQ(SIFT_DESCR_WIDTH, 4) 
        << "SIFT descriptor width should be 4";
    
    EXPECT_EQ(SIFT_DESCR_HIST_BINS, 8)
        << "SIFT descriptor histogram bins should be 8";
    
    EXPECT_EQ(SIFT_DESCRIPTOR_SIZE, 128)
        << "SIFT descriptor size should be 4*4*8 = 128";
}

TEST_F(ConstantsTest, RGBSIFTConstants) {
    using namespace thesis_project::constants;
    
    EXPECT_EQ(RGB_SIFT_DESCRIPTOR_SIZE, 3 * SIFT_DESCRIPTOR_SIZE)
        << "RGB SIFT should be 3 times regular SIFT descriptor size";
    
    EXPECT_EQ(RGB_SIFT_DESCRIPTOR_SIZE, 384)
        << "RGB SIFT descriptor size should be 3*128 = 384";
}

TEST_F(ConstantsTest, ConstantsRelationships) {
    using namespace thesis_project::constants;
    
    // Test mathematical relationships
    EXPECT_EQ(SIFT_DESCRIPTOR_SIZE, SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS)
        << "SIFT descriptor size calculation should be correct";
    
    EXPECT_GT(RGB_SIFT_DESCRIPTOR_SIZE, SIFT_DESCRIPTOR_SIZE)
        << "RGB SIFT descriptor should be larger than regular SIFT";
    
    EXPECT_EQ(RGB_SIFT_DESCRIPTOR_SIZE % SIFT_DESCRIPTOR_SIZE, 0)
        << "RGB SIFT size should be evenly divisible by SIFT size";
}

// Test that constants are compile-time constants
TEST_F(ConstantsTest, CompileTimeConstants) {
    // These should compile as constexpr
    constexpr int sift_size = thesis_project::constants::SIFT_DESCRIPTOR_SIZE;
    constexpr int rgb_sift_size = thesis_project::constants::RGB_SIFT_DESCRIPTOR_SIZE;
    
    EXPECT_EQ(sift_size, 128) << "SIFT size should be compile-time constant";
    EXPECT_EQ(rgb_sift_size, 384) << "RGB SIFT size should be compile-time constant";
}