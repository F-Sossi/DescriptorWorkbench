#pragma once

#include <string>

namespace thesis_project::cli::experiment_runner_cli {

void printUsage(const std::string& binaryName);

// Parse argv for config path, handling --help/-h and argument count validation.
// Returns empty string on help or validation failure (after printing usage).
std::string parseArgs(int argc, char** argv);

} // namespace thesis_project::cli::experiment_runner_cli
