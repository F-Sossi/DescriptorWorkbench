#include "thesis_project/database/DatabaseManager.hpp"
#include "cli/keypoint_manager/commands/list_commands.hpp"
#include "cli/keypoint_manager/commands/import_export.hpp"
#include "cli/keypoint_manager/commands/generation_basic.hpp"
#include "cli/keypoint_manager/commands/generation_detector.hpp"
#include "cli/keypoint_manager/commands/subset_commands.hpp"
#include "cli/keypoint_manager/commands/intersection_commands.hpp"
#include "cli/keypoint_manager/commands/non_overlapping.hpp"
#include "cli/keypoint_manager/commands/subset_commands.hpp"
#include "src/core/keypoints/locked_in_keypoints.hpp"
#include "src/core/keypoints/KeypointGeneratorFactory.hpp"
#include "src/core/processing/processor_utils.hpp"
#include "src/core/utils/PythonEnvironment.hpp"
#include "thesis_project/logging.hpp"
#include "thesis_project/types.hpp"
#include <iostream>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <random>

namespace cv {
    class KeyPoint;
}

static void printUsage(const std::string& binaryName) {
    std::cout << "Usage: " << binaryName << " <command> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help                                Show this help message and exit" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  Keypoint Generation:" << std::endl;
    std::cout << "    generate-projected <data_folder> [name]   - Generate keypoints using homography projection (controlled)" << std::endl;
    std::cout << "    generate-independent <data_folder> [name] - Generate keypoints using independent detection (realistic)" << std::endl;
    std::cout << "    generate <data_folder>                    - Legacy: Generate homography projected keypoints" << std::endl;
    std::cout << "  Advanced Detector Generation:" << std::endl;
    std::cout << "    generate-detector <data_folder> <detector> [name]                    - Generate keypoints using specific detector (sift|harris|orb)" << std::endl;
    std::cout << "    generate-non-overlapping <data_folder> <detector> <min_distance> [name] - Generate non-overlapping keypoints" << std::endl;
    std::cout << "    generate-kornia-keynet <data_folder> [set_name] [max_kp] [device] [--mode independent|projected] [--overwrite]" << std::endl;
    std::cout << "                         Run Kornia KeyNet detector via Python (independent or homography projected)" << std::endl;
    std::cout << "  Subset Generation:" << std::endl;
    std::cout << "    generate-random-subset <source_set> <target_count> <output_name> [--seed N]" << std::endl;
    std::cout << "                         Generate random subset of keypoints from existing set" << std::endl;
    std::cout << "    generate-top-n <source_set> <target_count> --by <property> --out <output_name>" << std::endl;
    std::cout << "                         Generate top-N keypoints by property (response|size|octave)" << std::endl;
    std::cout << "    generate-spatial-subset <source_set> <target_count> <min_distance> <output_name>" << std::endl;
    std::cout << "                         Generate spatially filtered subset with minimum spacing" << std::endl;
    std::cout << "  Keypoint Set Operations:" << std::endl;
    std::cout << "    build-intersection --source-a <set> --source-b <set> --out-a <set> --out-b <set> [--tolerance px] [--overwrite]" << std::endl;
    std::cout << "                         Create paired subsets where two detectors agree spatially" << std::endl;
    std::cout << "  Import/Export:" << std::endl;
    std::cout << "    import-csv <csv_folder> [set_name]        - Import keypoints from CSV files" << std::endl;
    std::cout << "    export-csv <output_folder> [set_id]       - Export keypoints from DB to CSV" << std::endl;
    std::cout << "  Information:" << std::endl;
    std::cout << "    list-sets                                 - List all available keypoint sets" << std::endl;
    std::cout << "    list-scenes [set_id]                      - List scenes in database (optionally for specific set)" << std::endl;
    std::cout << "    count <scene> <image> [set_id]            - Count keypoints for specific scene/image" << std::endl;
    std::cout << "    list-detectors                            - List supported detector types" << std::endl;
}

using namespace thesis_project;

/**
 * @brief CLI tool for managing locked-in keypoints in the database
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    if (command == "--help" || command == "-h") {
        printUsage(argv[0]);
        return 0;
    }

    // Initialize database
    database::DatabaseManager db("experiments.db", true);
    if (!db.isEnabled()) {
        std::cerr << "❌ Failed to connect to database" << std::endl;
        return 1;
    }
    
    // Optimize database for bulk operations
    if (!db.optimizeForBulkOperations()) {
        std::cerr << "Warning: Failed to apply database optimizations" << std::endl;
    }

    if (command == "import-csv") {
        return thesis_project::cli::keypoint_commands::importCsv(db, argc, argv);

    } else if (command == "generate-projected") {
        return thesis_project::cli::keypoint_commands::generateProjected(db, argc, argv);

    } else if (command == "generate-independent") {
        return thesis_project::cli::keypoint_commands::generateIndependent(db, argc, argv);

    } else if (command == "generate") {
        return thesis_project::cli::keypoint_commands::generateLegacy(db, argc, argv);

    } else if (command == "export-csv") {
        return thesis_project::cli::keypoint_commands::exportCsv(db, argc, argv);

    } else if (command == "list-sets") {
        return thesis_project::cli::keypoint_commands::listSets(db);

    } else if (command == "list-scenes") {
        return thesis_project::cli::keypoint_commands::listScenes(db);

    } else if (command == "count") {
        return thesis_project::cli::keypoint_commands::countKeypoints(db, argc, argv);

    } else if (command == "generate-kornia-keynet") {
        return thesis_project::cli::keypoint_commands::generateKorniaKeynet(db, argc, argv);

    } else if (command == "generate-detector") {
        return thesis_project::cli::keypoint_commands::generateDetector(db, argc, argv);

    } else if (command == "generate-random-subset") {
        return thesis_project::cli::keypoint_commands::generateRandomSubset(db, argc, argv);

    } else if (command == "generate-top-n") {
        return thesis_project::cli::keypoint_commands::generateTopNSubset(db, argc, argv);

    } else if (command == "generate-spatial-subset") {
        return thesis_project::cli::keypoint_commands::generateSpatialSubset(db, argc, argv);

    } else if (command == "build-intersection") {
        return thesis_project::cli::keypoint_commands::buildIntersection(db, argc, argv);

    } else if (command == "generate-non-overlapping") {
        return thesis_project::cli::keypoint_commands::generateNonOverlapping(db, argc, argv);

    } else if (command == "list-detectors") {
        return thesis_project::cli::keypoint_commands::listDetectors();

    } else {
        std::cerr << "❌ Unknown command: " << command << std::endl;
        std::cerr << "Run without arguments to see available commands." << std::endl;
        return 1;
    }

    return 0;
}
