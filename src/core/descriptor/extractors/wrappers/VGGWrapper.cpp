#include "VGGWrapper.hpp"
#include <stdexcept>

namespace thesis_project {
namespace wrappers {

VGGWrapper::VGGWrapper() {
#ifdef HAVE_OPENCV_XFEATURES2D
    extractor_ = cv::xfeatures2d::VGG::create();
#else
    extractor_.release();
#endif
}

VGGWrapper::VGGWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
#ifdef HAVE_OPENCV_XFEATURES2D
    extractor_ = cv::xfeatures2d::VGG::create();
#else
    extractor_.release();
#endif
}

cv::Mat VGGWrapper::extract(const cv::Mat& image,
                            const std::vector<cv::KeyPoint>& keypoints,
                            const DescriptorParams& /*params*/) {
    if (extractor_.empty()) {
        throw std::runtime_error("OpenCV xfeatures2d::VGG not available. Rebuild OpenCV with contrib.");
    }
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> kps = keypoints;
    extractor_->compute(image, kps, descriptors);
    return descriptors;
}

} // namespace wrappers
} // namespace thesis_project

