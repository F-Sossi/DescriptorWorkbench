#include "VSIFTWrapper.hpp"
#include <sstream>

namespace thesis_project {
namespace wrappers {

VSIFTWrapper::VSIFTWrapper() {
    vsift_ = VanillaSIFT::create();
}

VSIFTWrapper::VSIFTWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    vsift_ = VanillaSIFT::create();
}

cv::Mat VSIFTWrapper::extract(const cv::Mat& image,
                             const std::vector<cv::KeyPoint>& keypoints,
                             const DescriptorParams& /*params*/) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;
    vsift_->compute(image, mutable_keypoints, descriptors);
    return descriptors;
}

std::string VSIFTWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "vSIFT Wrapper Configuration:\n";
    ss << "  Descriptor size: " << descriptorSize() << "\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace wrappers
} // namespace thesis_project

