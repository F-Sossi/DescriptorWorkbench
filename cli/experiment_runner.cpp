#include "cli/experiment_runner/cli_args.hpp"
#include "cli/experiment_runner/bootstrap.hpp"
#include "cli/experiment_runner/descriptor_run.hpp"
#include "cli/experiment_runner/helpers.hpp"
#include "cli/experiment_runner/keypoint_resolution.hpp"
#include "cli/experiment_runner/result_recording.hpp"
#include "thesis_project/logging.hpp"
#include "thesis_project/types.hpp"
#include <exception>
#include <string>

namespace cli_helpers = thesis_project::cli::experiment_runner_helpers;

int main(const int argc, char** argv) {
    const std::string config_path = thesis_project::cli::experiment_runner_cli::parseArgs(argc, argv);
    if (config_path.empty()) {
        return 0;
    }

    try {
        LOG_INFO("Loading experiment configuration from: " + config_path);
        auto bootstrap = thesis_project::cli::experiment_runner_bootstrap::loadConfigAndDatabase(config_path);
        const auto& yaml_config = bootstrap.config;
        const auto db_ptr = std::move(bootstrap.db);
        const bool db_enabled = bootstrap.db_enabled;

        LOG_INFO("Experiment: " + yaml_config.experiment.name);
        LOG_INFO("Description: " + yaml_config.experiment.description);
        LOG_INFO("Dataset: " + yaml_config.dataset.path);
        LOG_INFO("Descriptors: " + std::to_string(yaml_config.descriptors.size()));

        for (size_t i = 0; i < yaml_config.descriptors.size(); ++i) {
            const auto& desc_config = yaml_config.descriptors[i];

            LOG_INFO("Running experiment with descriptor: " + desc_config.name);
            const std::string execution_device = cli_helpers::normalizeDeviceString(desc_config.params.device);

            const int descriptor_keypoint_set_id =
                thesis_project::cli::experiment_runner_keypoints::resolveKeypointSetForLoading(
                    db_enabled ? db_ptr.get() : nullptr,
                    yaml_config,
                    desc_config
                );

            auto run_result = thesis_project::cli::experiment_runner_descriptor::runDescriptor(
                yaml_config,
                desc_config,
                db_enabled ? db_ptr.get() : nullptr,
                descriptor_keypoint_set_id,
                execution_device);

            if (run_result.experiment_id != -1 && db_enabled && db_ptr) {
                thesis_project::cli::experiment_runner_results::recordResults(
                    *db_ptr,
                    run_result.experiment_id,
                    desc_config.name,
                    yaml_config.dataset.path,
                    run_result.metrics,
                    run_result.profile,
                    execution_device,
                    yaml_config.experiment.name);
            }

            if (run_result.metrics.success) {
                LOG_INFO("Completed descriptor: " + desc_config.name);
            } else {
                LOG_ERROR("Failed descriptor: " + desc_config.name);
            }
        }

        LOG_INFO("Experiment completed: " + yaml_config.experiment.name);
        LOG_INFO("Experiment results saved to database");

        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Experiment failed: " + std::string(e.what()));
        return 1;
    }
}
