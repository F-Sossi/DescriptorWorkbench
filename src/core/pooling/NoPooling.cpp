#include "NoPooling.hpp"
#include "src/interfaces/IDescriptorExtractor.hpp"

namespace thesis_project::pooling {

cv::Mat NoPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    // Convert DescriptorConfig to DescriptorParams to pass device setting
    thesis_project::DescriptorParams params = descCfg.params;
    return extractor.extract(image, keypoints, params);
}

} // namespace thesis_project::pooling
