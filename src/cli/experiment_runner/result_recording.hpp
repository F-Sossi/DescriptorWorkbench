#pragma once

#include "src/core/metrics/ExperimentMetrics.hpp"
#include "cli/experiment_runner/types.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <string>

namespace thesis_project::cli::experiment_runner_results {

void recordResults(
    thesis_project::database::DatabaseManager& db,
    int experiment_id,
    const std::string& descriptor_name,
    const std::string& dataset_path,
    const ::ExperimentMetrics& metrics,
    const ProfilingSummary& profile,
    const std::string& execution_device,
    const std::string& experiment_name);

} // namespace thesis_project::cli::experiment_runner_results
