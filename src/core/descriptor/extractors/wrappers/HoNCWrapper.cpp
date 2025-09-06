#include "HoNCWrapper.hpp"
#include <sstream>

namespace thesis_project {
namespace wrappers {

HoNCWrapper::HoNCWrapper() {
    honc_ = std::make_unique<HoNC>();
}

HoNCWrapper::HoNCWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    honc_ = std::make_unique<HoNC>();
}

cv::Mat HoNCWrapper::extract(const cv::Mat& image,
                            const std::vector<cv::KeyPoint>& keypoints,
                            const DescriptorParams& /*params*/) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;
    honc_->compute(image, mutable_keypoints, descriptors);
    return descriptors;
}

std::string HoNCWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "HoNC Wrapper Configuration:\n";
    ss << "  Descriptor size: " << descriptorSize() << "\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace wrappers
} // namespace thesis_project

