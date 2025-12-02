#pragma once

#include "src/core/config/ExperimentConfig.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <memory>
#include <string>

namespace thesis_project::cli::experiment_runner_bootstrap {

struct BootstrapResult {
    config::ExperimentConfig config;
    std::unique_ptr<thesis_project::database::DatabaseManager> db;
    bool db_enabled = false;
};

// Load YAML config, initialize database (respecting YAML connection string),
// and resolve primary keypoint set ID when using inheritance mode.
BootstrapResult loadConfigAndDatabase(const std::string& config_path);

} // namespace thesis_project::cli::experiment_runner_bootstrap
