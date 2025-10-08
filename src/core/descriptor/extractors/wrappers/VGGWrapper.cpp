#include "VGGWrapper.hpp"
#include <stdexcept>

namespace thesis_project::wrappers {

    VGGWrapper::VGGWrapper() {
#ifdef HAVE_OPENCV_XFEATURES2D
        // Create with default parameters (will be updated in extract() if needed)
        extractor_ = cv::xfeatures2d::VGG::create();
#else
        extractor_.release();
#endif
    }

    cv::Mat VGGWrapper::extract(const cv::Mat& image,
                                const std::vector<cv::KeyPoint>& keypoints,
                                const DescriptorParams& params) {
        if (extractor_.empty()) {
            throw std::runtime_error("OpenCV xfeatures2d::VGG not available. Rebuild OpenCV with contrib.");
        }

#ifdef HAVE_OPENCV_XFEATURES2D
        // Recreate VGG extractor with explicit parameters to ensure consistency
        extractor_ = cv::xfeatures2d::VGG::create(
            params.vgg_desc_type,           // Descriptor type (120, 80, 64, or 48 dims)
            params.vgg_isigma,              // Gaussian sigma
            params.vgg_img_normalize,       // Image normalization
            params.vgg_use_scale_orientation, // Use keypoint scale/orientation
            params.vgg_scale_factor,        // Sampling window scale factor
            params.vgg_dsc_normalize        // Descriptor normalization
        );
#endif

        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;
        extractor_->compute(image, kps, descriptors);
        return descriptors;
    }

}
