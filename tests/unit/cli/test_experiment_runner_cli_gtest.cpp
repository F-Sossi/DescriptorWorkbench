#include <gtest/gtest.h>
#include "cli/experiment_runner/cli_args.hpp"

namespace cli = thesis_project::cli::experiment_runner_cli;

namespace thesis_project::cli::experiment_runner_helpers {
std::string normalizeDeviceString(const std::string& raw_device);
}

TEST(ExperimentRunnerCliArgs, AcceptsSingleConfig) {
    const char* argv[] = {"experiment_runner", "config.yaml"};
    std::string config = cli::parseArgs(2, const_cast<char**>(argv));
    EXPECT_EQ(config, "config.yaml");
}

TEST(ExperimentRunnerCliArgs, RejectsHelp) {
    const char* argv[] = {"experiment_runner", "--help"};
    std::string config = cli::parseArgs(2, const_cast<char**>(argv));
    EXPECT_TRUE(config.empty());
}

TEST(ExperimentRunnerCliArgs, RejectsExtraArgs) {
    const char* argv[] = {"experiment_runner", "config.yaml", "extra"};
    std::string config = cli::parseArgs(3, const_cast<char**>(argv));
    EXPECT_TRUE(config.empty());
}

TEST(ExperimentRunnerHelpers, NormalizeDeviceStringVariants) {
    EXPECT_EQ(thesis_project::cli::experiment_runner_helpers::normalizeDeviceString("GPU"), "cuda");
    EXPECT_EQ(thesis_project::cli::experiment_runner_helpers::normalizeDeviceString("gpu+cpu"), "mixed");
    EXPECT_EQ(thesis_project::cli::experiment_runner_helpers::normalizeDeviceString(""), "auto");
    EXPECT_EQ(thesis_project::cli::experiment_runner_helpers::normalizeDeviceString("cpu"), "cpu");
    EXPECT_EQ(thesis_project::cli::experiment_runner_helpers::normalizeDeviceString("nonsense"), "nonsense");
}
