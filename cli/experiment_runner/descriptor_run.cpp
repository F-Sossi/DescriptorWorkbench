#include "descriptor_run.hpp"
#include "cli/experiment_runner/helpers.hpp"
#include "cli/experiment_runner/processing.hpp"
#include "src/core/config/ExperimentConfig.hpp"
#include "thesis_project/logging.hpp"
#include <chrono>

namespace thesis_project::cli::experiment_runner_descriptor {

DescriptorRunResult runDescriptor(
    const config::ExperimentConfig& yaml_config,
    const config::ExperimentConfig::DescriptorConfig& desc_config,
    thesis_project::database::DatabaseManager* db_ptr,
    const int descriptor_keypoint_set_id,
    const std::string& execution_device) {

    DescriptorRunResult result;
    const bool db_enabled = db_ptr && db_ptr->isEnabled();

    thesis_project::database::ExperimentConfig dbConfig;
    int experiment_id = -1;

    if (db_enabled) {
        dbConfig.descriptor_type = desc_config.name;
        dbConfig.dataset_path = yaml_config.dataset.path;
        dbConfig.pooling_strategy = toString(desc_config.params.pooling);
        dbConfig.similarity_threshold = yaml_config.evaluation.params.match_threshold;
        dbConfig.max_features = yaml_config.keypoints.params.max_features;
        dbConfig.parameters["experiment_name"] = yaml_config.experiment.name;
        dbConfig.parameters["descriptor_type"] = toString(desc_config.type);
        dbConfig.parameters["pooling_strategy"] = toString(desc_config.params.pooling);
        dbConfig.parameters["norm_type"] = std::to_string(desc_config.params.norm_type);
        dbConfig.execution_device = execution_device;
        dbConfig.descriptor_dimension = 0;
        dbConfig.parameters["execution_device"] = execution_device;
        dbConfig.keypoint_set_id = descriptor_keypoint_set_id;
        dbConfig.keypoint_source = toString(yaml_config.keypoints.params.source);
        experiment_id = db_ptr->recordConfiguration(dbConfig);
    }

    const auto start_time = std::chrono::high_resolution_clock::now();

    result.metrics = thesis_project::cli::experiment_runner_processing::processDirectoryNew(
        yaml_config,
        desc_config,
        db_enabled ? db_ptr : nullptr,
        experiment_id,
        descriptor_keypoint_set_id,
        result.profile);

    const auto end_time = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.experiment_id = experiment_id;

    if (db_enabled && experiment_id != -1) {
        db_ptr->updateExperimentDescriptorMetadata(
            experiment_id,
            result.profile.descriptor_dimension,
            execution_device);
    }

    return result;
}

} // namespace thesis_project::cli::experiment_runner_descriptor
