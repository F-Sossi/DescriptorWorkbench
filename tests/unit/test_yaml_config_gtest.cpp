#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <cctype>

// Configuration concepts from Stage 2
enum class DescriptorType { SIFT, RGBSIFT, vSIFT, HoNC };
enum class PoolingStrategy { NONE, DOMAIN_SIZE_POOLING, STACKING };

std::string toString(DescriptorType type) {
    switch (type) {
        case DescriptorType::SIFT: return "sift";
        case DescriptorType::RGBSIFT: return "rgbsift";
        case DescriptorType::vSIFT: return "vsift";
        case DescriptorType::HoNC: return "honc";
        default: return "unknown";
    }
}

namespace {

std::filesystem::path repoRoot() {
    auto source_path = std::filesystem::path(__FILE__);
    return source_path.parent_path().parent_path().parent_path();
}

std::filesystem::path requiredConfigListPath() {
    return repoRoot() / "tests" / "unit" / "config" / "required_configs.txt";
}

std::string trim(const std::string& input) {
    auto first = std::find_if_not(input.begin(), input.end(), [](unsigned char ch) { return std::isspace(ch); });
    auto last = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char ch) { return std::isspace(ch); }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

std::vector<std::string> loadRequiredConfigList() {
    std::vector<std::string> configs;
    std::ifstream list_file(requiredConfigListPath());
    if (!list_file.good()) {
        return configs;
    }

    std::string line;
    while (std::getline(list_file, line)) {
        auto cleaned = trim(line);
        if (cleaned.empty() || cleaned[0] == '#') {
            continue;
        }
        configs.push_back(cleaned);
    }
    return configs;
}

const std::vector<std::string>& cachedRequiredConfigs() {
    static const std::vector<std::string> configs = loadRequiredConfigList();
    return configs;
}

std::vector<std::pair<std::string, std::string>> baselineSectionPairs(const std::vector<std::string>& sections) {
    std::vector<std::pair<std::string, std::string>> pairs;
    const auto& configs = cachedRequiredConfigs();
    if (configs.empty()) {
        return pairs;
    }
    const auto& baseline = configs.front();
    for (const auto& section : sections) {
        pairs.emplace_back(baseline, section);
    }
    return pairs;
}

} // namespace

class YAMLConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_files = cachedRequiredConfigs();
        if (config_files.empty()) {
            GTEST_SKIP() << "No configuration paths listed in "
                         << requiredConfigListPath().string();
        }

        required_sections = {
            "experiment:", "dataset:", "keypoints:", "descriptors:",
            "evaluation:", "database:"
        };
        baseline_config = config_files.front();
    }
    
    std::vector<std::string> config_files;
    std::vector<std::string> required_sections;
    std::string baseline_config;
};

TEST_F(YAMLConfigTest, ConfigurationFilesExist) {
    int found_configs = 0;
    std::vector<std::string> missing_files;
    
    for (const auto& file : config_files) {
        std::ifstream test_file(file);
        if (test_file.good()) {
            found_configs++;
        } else {
            missing_files.push_back(file);
        }
        test_file.close();
    }
    
    // Report which files are missing if any
    if (!missing_files.empty()) {
        std::string missing_list = "Missing files: ";
        for (size_t i = 0; i < missing_files.size(); ++i) {
            missing_list += missing_files[i];
            if (i < missing_files.size() - 1) missing_list += ", ";
        }
        FAIL() << missing_list;
    }
    
    EXPECT_EQ(found_configs, static_cast<int>(config_files.size()))
        << "Found " << found_configs << "/" << config_files.size() << " configuration files";
}

TEST_F(YAMLConfigTest, SIFTBaselineStructure) {
    std::ifstream config_file(baseline_config);
    ASSERT_TRUE(config_file.good()) 
        << "Baseline configuration file must exist for structure validation: " << baseline_config;
    
    std::string content((std::istreambuf_iterator<char>(config_file)),
                        std::istreambuf_iterator<char>());
    config_file.close();
    
    std::vector<std::string> missing_sections;
    int found_sections = 0;
    
    for (const auto& section : required_sections) {
        if (content.find(section) != std::string::npos) {
            found_sections++;
        } else {
            missing_sections.push_back(section);
        }
    }
    
    // Report which sections are missing if any
    if (!missing_sections.empty()) {
        std::string missing_list = "Missing sections: ";
        for (size_t i = 0; i < missing_sections.size(); ++i) {
            missing_list += missing_sections[i];
            if (i < missing_sections.size() - 1) missing_list += ", ";
        }
        FAIL() << missing_list;
    }
    
    EXPECT_EQ(found_sections, static_cast<int>(required_sections.size()))
        << "YAML structure validation should find all required sections";
}

TEST_F(YAMLConfigTest, DescriptorTypeMapping) {
    // Test basic descriptor type to string mappings
    EXPECT_EQ(toString(DescriptorType::SIFT), "sift") 
        << "SIFT descriptor type should map to 'sift'";
    
    EXPECT_EQ(toString(DescriptorType::RGBSIFT), "rgbsift")
        << "RGBSIFT descriptor type should map to 'rgbsift'";
    
    EXPECT_EQ(toString(DescriptorType::vSIFT), "vsift")
        << "vSIFT descriptor type should map to 'vsift'";
    
    EXPECT_EQ(toString(DescriptorType::HoNC), "honc")
        << "HoNC descriptor type should map to 'honc'";
}

TEST_F(YAMLConfigTest, PoolingStrategyEnums) {
    // Test that pooling strategy enum values are defined
    PoolingStrategy none_strategy = PoolingStrategy::NONE;
    PoolingStrategy dsp_strategy = PoolingStrategy::DOMAIN_SIZE_POOLING;
    PoolingStrategy stacking_strategy = PoolingStrategy::STACKING;
    
    // Test that the enums can be compared
    EXPECT_NE(none_strategy, dsp_strategy) 
        << "NONE and DOMAIN_SIZE_POOLING should be different strategies";
    
    EXPECT_NE(dsp_strategy, stacking_strategy)
        << "DOMAIN_SIZE_POOLING and STACKING should be different strategies";
    
    EXPECT_NE(none_strategy, stacking_strategy)
        << "NONE and STACKING should be different strategies";
}

// Parameterized test for checking each config file individually
class ConfigFileTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(ConfigFileTest, FileExists) {
    const std::string& config_file = GetParam();
    std::ifstream test_file(config_file);
    
    EXPECT_TRUE(test_file.good()) 
        << "Configuration file should exist: " << config_file;
    
    if (test_file.good()) {
        test_file.close();
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigFiles,
    ConfigFileTest,
    ::testing::ValuesIn(cachedRequiredConfigs())
);

// Test fixture for testing multiple YAML structure elements
class YAMLStructureTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {
protected:
    void SetUp() override {
        const auto& param = GetParam();
        config_file_path = param.first;
        required_section = param.second;
    }
    
    std::string config_file_path;
    std::string required_section;
};

TEST_P(YAMLStructureTest, SectionExists) {
    std::ifstream config_file(config_file_path);
    if (!config_file.good()) {
        GTEST_SKIP() << "Skipping structure test - config file not found: " << config_file_path;
    }
    
    std::string content((std::istreambuf_iterator<char>(config_file)),
                        std::istreambuf_iterator<char>());
    config_file.close();
    
    EXPECT_NE(content.find(required_section), std::string::npos)
        << "Section '" << required_section << "' should exist in " << config_file_path;
}

INSTANTIATE_TEST_SUITE_P(
    SIFTBaselineSections,
    YAMLStructureTest,
    ::testing::ValuesIn(
        baselineSectionPairs({
            "experiment:",
            "dataset:",
            "keypoints:",
            "descriptors:",
            "evaluation:",
            "database:"
        })
    )
);
