#include "bootstrap.hpp"
#include "src/core/config/YAMLConfigLoader.hpp"
#include "thesis_project/logging.hpp"

namespace thesis_project::cli::experiment_runner_bootstrap {

BootstrapResult loadConfigAndDatabase(const std::string& config_path) {
    BootstrapResult result;
    result.config = config::YAMLConfigLoader::loadFromFile(config_path);

    auto normalizeDbPath = [](std::string s) {
        if (const std::string prefix = "sqlite:///"; s.rfind(prefix, 0) == 0) {
            return s.substr(prefix.size());
        }
        return s;
    };

    std::string db_path = result.config.database.connection_string.empty()
                              ? std::string("experiments.db")
                              : normalizeDbPath(result.config.database.connection_string);

    result.db = std::make_unique<thesis_project::database::DatabaseManager>(db_path, true);
    result.db_enabled = result.db->isEnabled();

    if (result.db_enabled) {
        if (result.db->optimizeForBulkOperations()) {
            LOG_INFO("Bulk operations enabled");
        } else {
            LOG_INFO("Bulk operations disabled");
        }
        LOG_INFO("Database tracking enabled");
    } else {
        LOG_INFO("Database tracking disabled");
    }

    LOG_INFO("Keypoint assignment mode: " + toString(result.config.keypoints.assignment_mode));

    if (result.db_enabled &&
        result.config.keypoints.assignment_mode == KeypointAssignmentMode::INHERIT_FROM_PRIMARY &&
        !result.config.keypoints.params.keypoint_set_name.empty()) {

        const int resolved_set_id = result.db->getKeypointSetId(result.config.keypoints.params.keypoint_set_name);
        if (resolved_set_id == -1) {
            throw std::runtime_error("Keypoint set '" + result.config.keypoints.params.keypoint_set_name +
                                     "' not found in database");
        }
        result.config.keypoints.params.keypoint_set_id = resolved_set_id;
        LOG_INFO("Using keypoint set '" + result.config.keypoints.params.keypoint_set_name + "' (id=" +
                 std::to_string(resolved_set_id) + ") with source: " +
                 toString(result.config.keypoints.params.source));
    } else {
        result.config.keypoints.params.keypoint_set_id = -1;
        if (result.config.keypoints.assignment_mode == KeypointAssignmentMode::EXPLICIT_ONLY) {
            LOG_INFO("Primary keypoint set disabled (explicit assignment mode)");
        }
    }

    return result;
}

} // namespace thesis_project::cli::experiment_runner_bootstrap
