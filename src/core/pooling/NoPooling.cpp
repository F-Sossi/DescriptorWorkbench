#include "NoPooling.hpp"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/pooling/pooling_utils.hpp"
#include "include/thesis_project/types.hpp"

namespace thesis_project::pooling {

cv::Mat NoPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    // Convert DescriptorConfig to DescriptorParams to pass device setting
    thesis_project::DescriptorParams params = descCfg.params;
    cv::Mat descriptors = extractor.extract(image, keypoints, params);

    if (descriptors.empty()) {
        return descriptors;
    }

    // Apply optional normalization before pooling (interpreted as a pre-step even without pooling)
    if (params.normalize_before_pooling) {
        pooling::utils::normalizeRows(descriptors, params.norm_type);
    }

    // Rooting stage: for no-pooling scenario treat both BEFORE and AFTER as a single post-processing step.
    if (params.rooting_stage == thesis_project::RootingStage::R_BEFORE_POOLING ||
        params.rooting_stage == thesis_project::RootingStage::R_AFTER_POOLING) {
        pooling::utils::normalizeRows(descriptors, cv::NORM_L1);
        pooling::utils::applyRooting(descriptors);
    }

    if (params.normalize_after_pooling) {
        pooling::utils::normalizeRows(descriptors, params.norm_type);
    }

    return descriptors;
}

} // namespace thesis_project::pooling
