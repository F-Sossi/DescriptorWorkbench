#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <memory>
#include <exception>

// Test basic type system - same as original
enum class DescriptorType { SIFT, RGBSIFT, vSIFT, HoNC, NONE };
enum class PoolingStrategy { NONE, DOMAIN_SIZE_POOLING, STACKING };

// Mock interface for testing concepts - same as original
class IDescriptorExtractor {
public:
    virtual ~IDescriptorExtractor() = default;
    virtual std::string name() const = 0;
    virtual int descriptorSize() const = 0;
    virtual DescriptorType type() const = 0;
};

// Mock implementation - same as original
class MockSIFTExtractor : public IDescriptorExtractor {
public:
    std::string name() const override { return "MockSIFT"; }
    int descriptorSize() const override { return 128; }
    DescriptorType type() const override { return DescriptorType::SIFT; }
};

class SimpleInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Helper function for string conversion
        toString = [](DescriptorType t) -> std::string {
            switch (t) {
                case DescriptorType::SIFT: return "SIFT";
                case DescriptorType::RGBSIFT: return "RGBSIFT";
                case DescriptorType::vSIFT: return "vSIFT";
                case DescriptorType::HoNC: return "HoNC";
                case DescriptorType::NONE: return "NONE";
                default: return "UNKNOWN";
            }
        };
    }
    
    std::function<std::string(DescriptorType)> toString;
};

TEST_F(SimpleInterfaceTest, BasicTypeSystem) {
    // Test enum creation and assignment
    DescriptorType type = DescriptorType::RGBSIFT;
    PoolingStrategy strategy = PoolingStrategy::DOMAIN_SIZE_POOLING;
    
    EXPECT_EQ(static_cast<int>(type), 1) << "RGBSIFT should have value 1";
    EXPECT_EQ(static_cast<int>(strategy), 1) << "DOMAIN_SIZE_POOLING should have value 1";
    
    // Test that enums can be assigned and compared
    EXPECT_EQ(type, DescriptorType::RGBSIFT) << "Type assignment should work";
    EXPECT_EQ(strategy, PoolingStrategy::DOMAIN_SIZE_POOLING) << "Strategy assignment should work";
}

TEST_F(SimpleInterfaceTest, StringConversions) {
    std::vector<DescriptorType> types = {
        DescriptorType::SIFT,
        DescriptorType::RGBSIFT,
        DescriptorType::vSIFT,
        DescriptorType::HoNC
    };
    
    std::vector<std::string> expected_names = {"SIFT", "RGBSIFT", "vSIFT", "HoNC"};
    
    ASSERT_EQ(types.size(), expected_names.size()) << "Test setup should be consistent";
    
    for (size_t i = 0; i < types.size(); ++i) {
        EXPECT_EQ(toString(types[i]), expected_names[i])
            << "String conversion for type " << i << " should work";
    }
    
    // Test unknown type handling
    EXPECT_EQ(toString(static_cast<DescriptorType>(99)), "UNKNOWN")
        << "Unknown types should return UNKNOWN";
}

TEST_F(SimpleInterfaceTest, InterfaceConcept) {
    MockSIFTExtractor mock_extractor;
    
    // Test interface methods
    EXPECT_EQ(mock_extractor.name(), "MockSIFT") << "Mock extractor name should be correct";
    EXPECT_EQ(mock_extractor.descriptorSize(), 128) << "Mock extractor size should be 128";
    EXPECT_EQ(mock_extractor.type(), DescriptorType::SIFT) << "Mock extractor type should be SIFT";
    
    // Test polymorphism
    std::unique_ptr<IDescriptorExtractor> extractor = std::make_unique<MockSIFTExtractor>();
    EXPECT_EQ(extractor->name(), "MockSIFT") << "Polymorphic call should work";
    EXPECT_EQ(extractor->descriptorSize(), 128) << "Polymorphic size call should work";
    EXPECT_EQ(extractor->type(), DescriptorType::SIFT) << "Polymorphic type call should work";
}

TEST_F(SimpleInterfaceTest, FactoryConcept) {
    auto createMockExtractor = [](DescriptorType type) -> std::unique_ptr<IDescriptorExtractor> {
        switch (type) {
            case DescriptorType::SIFT:
                return std::make_unique<MockSIFTExtractor>();
            default:
                throw std::runtime_error("Unsupported type in mock factory");
        }
    };
    
    // Test successful creation
    EXPECT_NO_THROW({
        auto extractor = createMockExtractor(DescriptorType::SIFT);
        EXPECT_NE(extractor, nullptr) << "Factory should create valid extractor";
        EXPECT_EQ(extractor->name(), "MockSIFT") << "Factory-created extractor should work";
    }) << "Factory should successfully create SIFT extractor";
    
    // Test exception for unsupported type
    EXPECT_THROW({
        auto extractor = createMockExtractor(DescriptorType::HoNC);
    }, std::runtime_error) << "Factory should throw for unsupported types";
}

TEST_F(SimpleInterfaceTest, MemoryManagement) {
    // Test that smart pointers work correctly
    std::vector<std::unique_ptr<IDescriptorExtractor>> extractors;
    
    // Create multiple extractors
    for (int i = 0; i < 3; ++i) {
        extractors.push_back(std::make_unique<MockSIFTExtractor>());
    }
    
    EXPECT_EQ(extractors.size(), 3u) << "Should be able to store multiple extractors";
    
    // Test access through smart pointers
    for (auto& extractor : extractors) {
        EXPECT_NE(extractor, nullptr) << "Extractor should not be null";
        EXPECT_EQ(extractor->name(), "MockSIFT") << "All extractors should work correctly";
    }
    
    // Smart pointers will clean up automatically when going out of scope
    // This tests that the interface destructor works correctly
}

// Integration test that combines multiple concepts
TEST_F(SimpleInterfaceTest, IntegrationTest) {
    // Create a factory function
    auto factory = [](DescriptorType type) -> std::unique_ptr<IDescriptorExtractor> {
        if (type == DescriptorType::SIFT) {
            return std::make_unique<MockSIFTExtractor>();
        }
        return nullptr;
    };
    
    // Test workflow: type -> string -> factory -> interface
    std::vector<DescriptorType> test_types = {DescriptorType::SIFT};
    
    for (const auto& type : test_types) {
        // Convert to string
        std::string type_name = toString(type);
        EXPECT_FALSE(type_name.empty()) << "Type name should not be empty";
        
        // Create extractor via factory
        auto extractor = factory(type);
        ASSERT_NE(extractor, nullptr) << "Factory should create extractor for " << type_name;
        
        // Test interface methods
        EXPECT_EQ(extractor->type(), type) << "Created extractor should have correct type";
        EXPECT_GT(extractor->descriptorSize(), 0) << "Descriptor size should be positive";
        EXPECT_FALSE(extractor->name().empty()) << "Extractor name should not be empty";
    }
}