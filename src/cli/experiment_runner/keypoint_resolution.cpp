#include "keypoint_resolution.hpp"
#include "thesis_project/logging.hpp"

namespace thesis_project::cli::experiment_runner_keypoints {

int resolveKeypointSetForLoading(
    thesis_project::database::DatabaseManager* db,
    const thesis_project::config::ExperimentConfig& yaml_config,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& desc_config
) {
    auto resolveByName = [&](const std::string& set_name,
                             const std::string& context) -> int {
        if (set_name.empty()) {
            return -1;
        }
        if (!db || !db->isEnabled()) {
            LOG_WARNING(context + " requires keypoint set '" + set_name +
                        "' but database access is disabled");
            return -1;
        }

        const int resolved_id = db->getKeypointSetId(set_name);
        if (resolved_id < 0) {
            throw std::runtime_error(
                context + " references keypoint_set_name '" + set_name +
                "' which does not exist in the database. "
                "Generate it first using keypoint_manager."
            );
        }

        LOG_INFO(context + " loading keypoints from: " + set_name +
                 " (ID=" + std::to_string(resolved_id) + ")");
        return resolved_id;
    };

    if (!desc_config.keypoint_set_name.empty()) {
        return resolveByName(desc_config.keypoint_set_name,
                             "Descriptor '" + desc_config.name + "'");
    }

    if (desc_config.type == thesis_project::DescriptorType::COMPOSITE &&
        !desc_config.components.empty() &&
        !desc_config.components.front().keypoint_set_name.empty()) {

        LOG_INFO(
            "Composite descriptor '" + desc_config.name +
            "' will load scene keypoints from component[0] set '" +
            desc_config.components.front().keypoint_set_name + "'"
        );
        return resolveByName(
            desc_config.components.front().keypoint_set_name,
            "Composite '" + desc_config.name + "'"
        );
    }

    if (yaml_config.keypoints.assignment_mode == thesis_project::KeypointAssignmentMode::EXPLICIT_ONLY) {
        throw std::runtime_error(
            "Cannot resolve keypoint set for descriptor '" + desc_config.name + "': "
            "keypoints.keypoint_set_name is configured for explicit assignment, "
            "but neither the descriptor nor its components specify keypoint_set_name."
        );
    }

    if (yaml_config.keypoints.params.keypoint_set_id >= 0) {
        return yaml_config.keypoints.params.keypoint_set_id;
    }

    if (!yaml_config.keypoints.params.keypoint_set_name.empty()) {
        return resolveByName(
            yaml_config.keypoints.params.keypoint_set_name,
            "Descriptor '" + desc_config.name + "'"
        );
    }

    return -1;
}

} // namespace thesis_project::cli::experiment_runner_keypoints
