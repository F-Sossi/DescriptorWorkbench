#include "RGBSIFTWrapper.hpp"
#include <sstream>

namespace thesis_project {
namespace wrappers {

RGBSIFTWrapper::RGBSIFTWrapper() {
    rgbsift_ = std::make_unique<RGBSIFT>();
}

RGBSIFTWrapper::RGBSIFTWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    rgbsift_ = std::make_unique<RGBSIFT>();
}

cv::Mat RGBSIFTWrapper::extract(const cv::Mat& image,
                               const std::vector<cv::KeyPoint>& keypoints,
                               const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;
    rgbsift_->compute(image, mutable_keypoints, descriptors);
    return descriptors;
}

std::string RGBSIFTWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "RGBSIFT Wrapper Configuration:\n";
    ss << "  RGB SIFT descriptor\n";
    ss << "  Descriptor size: " << descriptorSize() << "\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace wrappers
} // namespace thesis_project
