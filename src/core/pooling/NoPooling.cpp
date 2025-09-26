#include "NoPooling.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/VanillaSIFT.h"
#include "src/interfaces/IDescriptorExtractor.hpp"

namespace thesis_project::pooling {

cv::Mat NoPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Ptr<cv::Feature2D>& detector,
    const experiment_config& config
) {
    cv::Mat descriptors;
    cv::Mat processedImage = image.clone();
    
    // Convert image to grayscale if needed for descriptor color space
    if (config.descriptorOptions.descriptorColorSpace == D_BW && image.channels() > 1) {
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
    }
    
    // Create a copy of keypoints to avoid modification
    std::vector<cv::KeyPoint> keypointsCopy = keypoints;
    
    // Handle VanillaSIFT vs standard Feature2D interface
    if (auto vanillaSift = std::dynamic_pointer_cast<VanillaSIFT>(detector)) {
        vanillaSift->compute(processedImage, keypointsCopy, descriptors);
    } else {
        detector->compute(processedImage, keypointsCopy, descriptors);
    }
    
    return descriptors;
}

// Legacy interface overload: delegate to extractor with default params
cv::Mat NoPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const experiment_config& /*config*/
) {
    // Legacy config doesn't have device settings - use default params
    thesis_project::DescriptorParams params; // defaults to device="auto"
    return extractor.extract(image, keypoints, params);
}

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
