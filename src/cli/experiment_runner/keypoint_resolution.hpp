#pragma once

#include "src/core/config/ExperimentConfig.hpp"
#include "thesis_project/database/DatabaseManager.hpp"

namespace thesis_project::cli::experiment_runner_keypoints {

int resolveKeypointSetForLoading(
    thesis_project::database::DatabaseManager* db,
    const thesis_project::config::ExperimentConfig& yaml_config,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& desc_config);

} // namespace thesis_project::cli::experiment_runner_keypoints
