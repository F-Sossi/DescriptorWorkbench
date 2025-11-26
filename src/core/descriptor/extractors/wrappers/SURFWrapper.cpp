#include "SURFWrapper.hpp"
#include <sstream>

namespace thesis_project::wrappers {

SURFWrapper::SURFWrapper() {
    // Create SURF with default parameters
    // hessianThreshold=400, nOctaves=4, nOctaveLayers=3, extended=false, upright=false
    surf_ = cv::xfeatures2d::SURF::create();
    extended_ = surf_ ? surf_->getExtended() : false;
}

cv::Mat SURFWrapper::extract(const cv::Mat& image,
                            const std::vector<cv::KeyPoint>& keypoints,
                            const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;

    // Reconfigure SURF if extended flag changed (64D vs 128D descriptors)
    if (const bool want_extended = params.surf_extended; !surf_ || want_extended != extended_) {
        surf_ = cv::xfeatures2d::SURF::create(
            400.0,   // hessianThreshold (default)
            4,       // nOctaves (default)
            3,       // nOctaveLayers (default)
            want_extended,
            false    // upright
        );
        extended_ = want_extended;
    }

    // SURF is similar to SIFT and should work well with SIFT keypoints
    // However, we'll apply some basic validation to ensure compatibility
    for (auto& kp : mutable_keypoints) {
        // Ensure valid size (SURF works with a range of sizes)
        if (kp.size <= 0.0f || std::isnan(kp.size) || std::isinf(kp.size)) {
            kp.size = 20.0f;  // Default SURF patch size
        }

        // Ensure valid angle
        if (std::isnan(kp.angle) || std::isinf(kp.angle)) {
            kp.angle = -1.0f;  // OpenCV uses -1 to indicate no orientation
        }

        // Ensure valid response
        if (std::isnan(kp.response) || std::isinf(kp.response)) {
            kp.response = 0.0f;
        }

        // SURF should handle octave values better than ORB, but let's be safe
        if (kp.octave < -1) {
            kp.octave = 0;
        }
    }

    try {
        surf_->compute(image, mutable_keypoints, descriptors);
    } catch (const cv::Exception& e) {
        // If SURF fails, return empty descriptors rather than crashing
        std::cerr << "SURF computation failed: " << e.what() << std::endl;
        std::cerr << "Image size: " << image.cols << "x" << image.rows
                  << ", keypoints: " << mutable_keypoints.size() << std::endl;
        return cv::Mat();
    }

    return descriptors;
}

std::string SURFWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "SURF Wrapper Configuration:\n";
    ss << "  Descriptor size: " << descriptorSize() << " floats\n";
    ss << "  Descriptor type: Float (CV_32F)\n";
    ss << "  Hessian threshold: " << (surf_ ? surf_->getHessianThreshold() : 400.0) << "\n";
    ss << "  Octaves: " << (surf_ ? surf_->getNOctaves() : 4) << "\n";
    ss << "  Octave layers: " << (surf_ ? surf_->getNOctaveLayers() : 3) << "\n";
    return ss.str();
}

} // namespace thesis_project::wrappers
