#pragma once

#include "src/core/config/ExperimentConfig.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "cli/experiment_runner/types.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <string>
#include <chrono>

namespace thesis_project::cli::experiment_runner_descriptor {

struct DescriptorRunResult {
    ::ExperimentMetrics metrics;
    ProfilingSummary profile;
    int experiment_id = -1;
    std::chrono::milliseconds duration{0};
};

DescriptorRunResult runDescriptor(
    const config::ExperimentConfig& yaml_config,
    const config::ExperimentConfig::DescriptorConfig& desc_config,
    thesis_project::database::DatabaseManager* db_ptr,
    int descriptor_keypoint_set_id,
    const std::string& execution_device);

} // namespace thesis_project::cli::experiment_runner_descriptor
