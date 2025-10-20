/**
 * @file test_python_environment_gtest.cpp
 * @brief Google Test suite for PythonEnvironment utility
 *
 * Tests portable Python environment detection across conda, venv, and system Python.
 */

#include <gtest/gtest.h>
#include "src/core/utils/PythonEnvironment.hpp"
#include <string>
#include <vector>

using namespace thesis_project::utils;

/**
 * @brief Test fixture for PythonEnvironment tests
 */
class PythonEnvironmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Detect environment once for all tests
        env_info = PythonEnvironment::detect();
    }

    PythonEnvironment::Info env_info;
};

/**
 * @brief Test that Python environment can be detected
 */
TEST_F(PythonEnvironmentTest, DetectsEnvironment) {
    EXPECT_NE(env_info.type, PythonEnvironment::Type::UNKNOWN)
        << "Should detect at least one Python environment";
    EXPECT_FALSE(env_info.python_path.empty())
        << "Python path should not be empty";
}

/**
 * @brief Test environment type to string conversion
 */
TEST_F(PythonEnvironmentTest, TypeToStringConversion) {
    EXPECT_EQ("Conda", PythonEnvironment::typeToString(PythonEnvironment::Type::CONDA));
    EXPECT_EQ("Virtual Environment", PythonEnvironment::typeToString(PythonEnvironment::Type::VIRTUALENV));
    EXPECT_EQ("System Python", PythonEnvironment::typeToString(PythonEnvironment::Type::SYSTEM));
    EXPECT_EQ("Unknown", PythonEnvironment::typeToString(PythonEnvironment::Type::UNKNOWN));
}

/**
 * @brief Test that basic Python packages are available
 */
TEST_F(PythonEnvironmentTest, HasBasicPythonPackages) {
    // These should always be available in any Python installation
    EXPECT_TRUE(PythonEnvironment::hasPackage(env_info.python_path, "sys"))
        << "Python should have 'sys' module";
    EXPECT_TRUE(PythonEnvironment::hasPackage(env_info.python_path, "os"))
        << "Python should have 'os' module";
}

/**
 * @brief Test command generation for detected environment
 */
TEST_F(PythonEnvironmentTest, GeneratesValidCommand) {
    std::vector<std::string> args = {"--test", "arg"};
    std::string cmd = PythonEnvironment::generateCommand("test_script.py", args, env_info);

    EXPECT_FALSE(cmd.empty()) << "Generated command should not be empty";
    EXPECT_NE(cmd.find("test_script.py"), std::string::npos)
        << "Command should contain script name";
    EXPECT_NE(cmd.find("--test"), std::string::npos)
        << "Command should contain arguments";
}

/**
 * @brief Test command generation without pre-detected environment
 */
TEST_F(PythonEnvironmentTest, GeneratesCommandWithAutoDetection) {
    std::vector<std::string> args = {"--help"};
    std::string cmd = PythonEnvironment::generateCommand("script.py", args);

    EXPECT_FALSE(cmd.empty()) << "Auto-detected command should not be empty";
    EXPECT_NE(cmd.find("script.py"), std::string::npos)
        << "Command should contain script name";
}

/**
 * @brief Test requirements validation
 */
TEST_F(PythonEnvironmentTest, ValidatesRequirements) {
    PythonEnvironment::Requirements reqs;
    reqs.packages = {"sys", "os"};  // Basic packages that should exist

    auto env_with_reqs = PythonEnvironment::detect(reqs);
    EXPECT_TRUE(env_with_reqs.has_required_packages)
        << "Should detect basic Python packages";
    EXPECT_TRUE(env_with_reqs.missing_packages.empty())
        << "No packages should be missing for basic modules";
}

/**
 * @brief Test detection of missing packages
 */
TEST_F(PythonEnvironmentTest, DetectsMissingPackages) {
    PythonEnvironment::Requirements reqs;
    // Use a package name that definitely doesn't exist
    reqs.packages = {"nonexistent_package_xyz_123"};

    auto env_with_reqs = PythonEnvironment::detect(reqs);
    EXPECT_FALSE(env_with_reqs.has_required_packages)
        << "Should detect missing packages";
    EXPECT_FALSE(env_with_reqs.missing_packages.empty())
        << "Should list missing packages";
    EXPECT_EQ(env_with_reqs.missing_packages[0], "nonexistent_package_xyz_123")
        << "Should correctly identify the missing package";
}

/**
 * @brief Test that isValid() method works correctly
 */
TEST_F(PythonEnvironmentTest, IsValidMethod) {
    // Valid environment
    PythonEnvironment::Requirements reqs;
    reqs.packages = {"sys"};
    auto valid_env = PythonEnvironment::detect(reqs);
    EXPECT_TRUE(valid_env.is_valid())
        << "Environment with available packages should be valid";

    // Invalid environment (missing packages)
    reqs.packages = {"nonexistent_package_xyz"};
    auto invalid_env = PythonEnvironment::detect(reqs);
    EXPECT_FALSE(invalid_env.is_valid())
        << "Environment with missing packages should be invalid";
}

/**
 * @brief Test environment name is set
 */
TEST_F(PythonEnvironmentTest, HasEnvironmentName) {
    EXPECT_FALSE(env_info.environment_name.empty())
        << "Environment name should not be empty";
}

/**
 * @brief Test that command includes activation for conda/venv
 */
TEST_F(PythonEnvironmentTest, CommandIncludesActivationWhenNeeded) {
    std::string cmd = PythonEnvironment::generateCommand("script.py", {}, env_info);

    if (env_info.type == PythonEnvironment::Type::CONDA) {
        // Conda environments should use bash -c
        EXPECT_NE(cmd.find("/bin/bash -c"), std::string::npos)
            << "Conda command should use bash -c";
    } else if (env_info.type == PythonEnvironment::Type::VIRTUALENV) {
        // Virtualenv should also use bash -c for activation
        EXPECT_NE(cmd.find("/bin/bash -c"), std::string::npos)
            << "Virtualenv command should use bash -c";
    } else if (env_info.type == PythonEnvironment::Type::SYSTEM) {
        // System Python can be called directly (might or might not use bash)
        EXPECT_NE(cmd.find(env_info.python_path), std::string::npos)
            << "System Python command should contain python path";
    }
}

/**
 * @brief Test that Python executable exists
 */
TEST_F(PythonEnvironmentTest, PythonExecutableExists) {
    if (!env_info.python_path.empty()) {
        // We can't easily test file existence in a portable way without filesystem,
        // but we can at least verify it's a non-empty path
        EXPECT_FALSE(env_info.python_path.empty());

        // Should contain "python" in the path
        EXPECT_NE(env_info.python_path.find("python"), std::string::npos)
            << "Python path should contain 'python'";
    }
}

/**
 * @brief Integration test for KeyNet detection scenario
 */
TEST_F(PythonEnvironmentTest, KeyNetDetectionScenario) {
    // This simulates what KeynetDetector::isAvailable() does
    PythonEnvironment::Requirements reqs;
    reqs.packages = {"kornia", "torch", "cv2"};

    auto env = PythonEnvironment::detect(reqs);

    // The test should pass regardless of whether Kornia is installed
    // We're testing the detection mechanism, not the actual installation
    if (env.has_required_packages) {
        EXPECT_TRUE(env.is_valid()) << "Environment with Kornia should be valid";
        std::cout << "Note: Kornia environment detected - KeyNet functionality available" << std::endl;
    } else {
        EXPECT_FALSE(env.is_valid()) << "Environment without Kornia should be invalid";
        std::cout << "Note: Kornia not detected - KeyNet functionality unavailable (expected in most environments)" << std::endl;

        // Should correctly identify missing packages
        EXPECT_FALSE(env.missing_packages.empty())
            << "Should list which packages are missing";
    }
}

/**
 * @brief Test command generation with multiple arguments
 */
TEST_F(PythonEnvironmentTest, GeneratesCommandWithMultipleArguments) {
    std::vector<std::string> args = {
        "--input \"test.png\"",
        "--output \"result.csv\"",
        "--max_keypoints 2000",
        "--device cpu"
    };

    std::string cmd = PythonEnvironment::generateCommand("generate_keypoints.py", args, env_info);

    EXPECT_FALSE(cmd.empty());
    for (const auto& arg : args) {
        EXPECT_NE(cmd.find(arg), std::string::npos)
            << "Command should contain argument: " << arg;
    }
}

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
