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

    // Optional debug output (can be enabled for troubleshooting)
    // Uncomment the following block to debug keypoint properties:
    /*
    if (!keypoints.empty()) {
        float min_size = keypoints[0].size, max_size = keypoints[0].size;
        int min_octave = keypoints[0].octave, max_octave = keypoints[0].octave;
        for (const auto& kp : keypoints) {
            min_size = std::min(min_size, kp.size);
            max_size = std::max(max_size, kp.size);
            min_octave = std::min(min_octave, kp.octave);
            max_octave = std::max(max_octave, kp.octave);
        }
        std::cout << "ORB: keypoints=" << keypoints.size()
                  << ", size=[" << min_size << "," << max_size << "]"
                  << ", octave=[" << min_octave << "," << max_octave << "]" << std::endl;
    }
    */

    // ORB has different requirements than SIFT for keypoint properties
    // Normalize all keypoint properties for ORB compatibility
    int normalized_count = 0;
    for (auto& kp : mutable_keypoints) {

        // ORB works best with moderate keypoint sizes (typically 5-50 pixels)
        // SIFT sizes can be much larger (up to 400+ pixels) which breaks ORB
        if (kp.size <= 0.0f || std::isnan(kp.size) || std::isinf(kp.size) || kp.size > 50.0f || kp.size < 5.0f) {
            kp.size = 31.0f;  // Standard ORB patch size
            normalized_count++;
        }

        // Ensure valid angle (ORB computes its own orientation)
        if (std::isnan(kp.angle) || std::isinf(kp.angle)) {
            kp.angle = -1.0f;  // OpenCV uses -1 to indicate no orientation
        }

        // Ensure valid response
        if (std::isnan(kp.response) || std::isinf(kp.response)) {
            kp.response = 0.0f;
        }

        // Ensure valid octave (ORB expects small octave values like 0,1,2,3)
        // SIFT octaves can be encoded with large numbers, normalize for ORB
        if (kp.octave < 0 || kp.octave > 10) {
            kp.octave = 0;  // Use base octave for all keypoints
        }
    }

    try {
        orb_->compute(image, mutable_keypoints, descriptors);
    } catch (const cv::Exception& e) {
        // If ORB fails, return empty descriptors rather than crashing
        std::cerr << "ORB computation failed: " << e.what() << std::endl;
        std::cerr << "Image size: " << image.cols << "x" << image.rows
                  << ", keypoints: " << mutable_keypoints.size()
                  << ", normalized: " << normalized_count << std::endl;
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