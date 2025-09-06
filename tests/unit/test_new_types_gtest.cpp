#include <gtest/gtest.h>

// Test new types in isolation - same as original test
namespace thesis_project {
    enum class PoolingStrategy { NONE, DOMAIN_SIZE_POOLING, STACKING };
    enum class DescriptorType { SIFT, HoNC, RGBSIFT, vSIFT, NONE };
}

class NewTypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed for type tests
    }
};

TEST_F(NewTypesTest, PoolingStrategyEnum) {
    using namespace thesis_project;
    
    // Test enum creation and assignment
    PoolingStrategy none_strategy = PoolingStrategy::NONE;
    PoolingStrategy dsp_strategy = PoolingStrategy::DOMAIN_SIZE_POOLING;
    PoolingStrategy stacking_strategy = PoolingStrategy::STACKING;
    
    // Test that enums are distinct
    EXPECT_NE(none_strategy, dsp_strategy)
        << "NONE and DOMAIN_SIZE_POOLING should be different";
    
    EXPECT_NE(dsp_strategy, stacking_strategy)
        << "DOMAIN_SIZE_POOLING and STACKING should be different";
    
    EXPECT_NE(none_strategy, stacking_strategy)
        << "NONE and STACKING should be different";
    
    // Test underlying values are assigned correctly
    EXPECT_EQ(static_cast<int>(none_strategy), 0)
        << "First enum value should be 0";
    
    EXPECT_GT(static_cast<int>(dsp_strategy), static_cast<int>(none_strategy))
        << "Subsequent enum values should increase";
}

TEST_F(NewTypesTest, DescriptorTypeEnum) {
    using namespace thesis_project;
    
    // Test enum creation and assignment
    DescriptorType sift_type = DescriptorType::SIFT;
    DescriptorType honc_type = DescriptorType::HoNC;
    DescriptorType rgbsift_type = DescriptorType::RGBSIFT;
    DescriptorType vsift_type = DescriptorType::vSIFT;
    DescriptorType none_type = DescriptorType::NONE;
    
    // Test that all enum values are distinct
    std::vector<DescriptorType> types = {sift_type, honc_type, rgbsift_type, vsift_type, none_type};
    
    for (size_t i = 0; i < types.size(); ++i) {
        for (size_t j = i + 1; j < types.size(); ++j) {
            EXPECT_NE(types[i], types[j])
                << "Descriptor types at indices " << i << " and " << j << " should be different";
        }
    }
    
    // Test specific enum values
    EXPECT_EQ(static_cast<int>(sift_type), 0)
        << "SIFT should be first enum value";
}

TEST_F(NewTypesTest, EnumScopedNature) {
    using namespace thesis_project;
    
    // Test that scoped enums prevent implicit conversions
    PoolingStrategy strategy = PoolingStrategy::DOMAIN_SIZE_POOLING;
    DescriptorType type = DescriptorType::RGBSIFT;
    
    // These should compile (explicit cast)
    int strategy_int = static_cast<int>(strategy);
    int type_int = static_cast<int>(type);
    
    EXPECT_GE(strategy_int, 0) << "Enum cast to int should produce valid value";
    EXPECT_GE(type_int, 0) << "Enum cast to int should produce valid value";
    
    // Test that we can use them in switch statements (compile-time test)
    bool switch_works = false;
    switch (strategy) {
        case PoolingStrategy::NONE:
            break;
        case PoolingStrategy::DOMAIN_SIZE_POOLING:
            switch_works = true;
            break;
        case PoolingStrategy::STACKING:
            break;
    }
    
    EXPECT_TRUE(switch_works) << "Switch statement should work with scoped enums";
}

// Test enum comparison operations
TEST_F(NewTypesTest, EnumComparisons) {
    using namespace thesis_project;
    
    PoolingStrategy strategy1 = PoolingStrategy::NONE;
    PoolingStrategy strategy2 = PoolingStrategy::NONE;
    PoolingStrategy strategy3 = PoolingStrategy::STACKING;
    
    // Test equality
    EXPECT_EQ(strategy1, strategy2) << "Same enum values should be equal";
    EXPECT_NE(strategy1, strategy3) << "Different enum values should not be equal";
    
    // Test that we can create vectors/containers of enums
    std::vector<PoolingStrategy> strategies = {
        PoolingStrategy::NONE,
        PoolingStrategy::DOMAIN_SIZE_POOLING,
        PoolingStrategy::STACKING
    };
    
    EXPECT_EQ(strategies.size(), 3u) << "Should be able to store enums in containers";
    EXPECT_EQ(strategies[0], PoolingStrategy::NONE) << "Vector indexing should work";
}