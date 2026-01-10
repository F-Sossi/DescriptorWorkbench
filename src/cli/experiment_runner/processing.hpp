#pragma once

#include "src/core/config/ExperimentConfig.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "cli/experiment_runner/types.hpp"
#include "thesis_project/database/DatabaseManager.hpp"

namespace thesis_project::cli::experiment_runner_processing {

::ExperimentMetrics processDirectoryNew(
    const thesis_project::config::ExperimentConfig& yaml_config,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& desc_config,
    thesis_project::database::DatabaseManager* db_ptr,
    int experiment_id,
    int descriptor_keypoint_set_id,
    ProfilingSummary& profiling);

} // namespace thesis_project::cli::experiment_runner_processing
