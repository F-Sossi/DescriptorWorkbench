#include "cli_args.hpp"
#include <iostream>

namespace thesis_project::cli::experiment_runner_cli {

void printUsage(const std::string& binaryName) {
    std::cout << "Usage: " << binaryName << " <config.yaml>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help        Show this help message and exit" << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << binaryName << " ../config/experiments/sift_baseline.yaml" << std::endl;
}

std::string parseArgs(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return {};
    }

    std::string firstArg = argv[1];
    if (firstArg == "--help" || firstArg == "-h") {
        printUsage(argv[0]);
        return {};
    }

    if (argc != 2) {
        printUsage(argv[0]);
        return {};
    }

    return argv[1];
}

} // namespace thesis_project::cli::experiment_runner_cli
