#include "SIFTWrapper.hpp"
#include <sstream>

namespace thesis_project::wrappers {

SIFTWrapper::SIFTWrapper() {
    sift_ = cv::SIFT::create();
}

SIFTWrapper::SIFTWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    // Initialize OpenCV SIFT with default parameters
    sift_ = cv::SIFT::create();
}

cv::Mat SIFTWrapper::extract(const cv::Mat& image,
                            const std::vector<cv::KeyPoint>& keypoints,
                            const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;
    sift_->compute(image, mutable_keypoints, descriptors);
    return descriptors;
}

std::string SIFTWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "SIFT Wrapper Configuration:\n";
    ss << "  OpenCV SIFT with default parameters\n";
    ss << "  Descriptor size: " << descriptorSize() << "\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace thesis_project::wrappers

