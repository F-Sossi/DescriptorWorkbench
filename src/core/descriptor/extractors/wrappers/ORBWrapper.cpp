#include "ORBWrapper.hpp"
#include <sstream>

namespace thesis_project::wrappers {

ORBWrapper::ORBWrapper() {
    orb_ = cv::ORB::create();
}

ORBWrapper::ORBWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    // Initialize OpenCV ORB with default parameters
    orb_ = cv::ORB::create();
}

cv::Mat ORBWrapper::extract(const cv::Mat& image,
                           const std::vector<cv::KeyPoint>& keypoints,
                           const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;

    // With spatial intersection and detector attribute hydration, keypoints should already
    // have appropriate ORB-specific parameters. Only basic safety checks needed.
    for (auto& kp : mutable_keypoints) {
        // Safety check: ensure no NaN/inf values that could crash ORB
        if (std::isnan(kp.size) || std::isinf(kp.size) || kp.size <= 0.0f) {
            kp.size = 31.0f;  // Standard ORB patch size fallback
        }
        if (std::isnan(kp.angle) || std::isinf(kp.angle)) {
            kp.angle = -1.0f;  // OpenCV uses -1 to indicate no orientation
        }
        if (std::isnan(kp.response) || std::isinf(kp.response)) {
            kp.response = 0.0f;
        }
    }

    try {
        orb_->compute(image, mutable_keypoints, descriptors);
    } catch (const cv::Exception& e) {
        // If ORB fails, return empty descriptors rather than crashing
        std::cerr << "ORB computation failed: " << e.what() << std::endl;
        std::cerr << "Image size: " << image.cols << "x" << image.rows
                  << ", keypoints: " << mutable_keypoints.size() << std::endl;
        return cv::Mat();
    }

    return descriptors;
}

std::string ORBWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "ORB Wrapper Configuration:\n";
    ss << "  OpenCV ORB with default parameters\n";
    ss << "  Descriptor size: " << descriptorSize() << " bytes\n";
    ss << "  Descriptor type: Binary (CV_8U)\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace thesis_project::wrappers