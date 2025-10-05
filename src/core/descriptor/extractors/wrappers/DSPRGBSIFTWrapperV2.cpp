#include "DSPRGBSIFTWrapperV2.hpp"

namespace thesis_project::wrappers {

    DSPRGBSIFTWrapperV2::DSPRGBSIFTWrapperV2()
        : wrapper_(cv::makePtr<DSPVanillaSIFTWrapper<cv::RGBSIFT>>()) {}

    cv::Mat DSPRGBSIFTWrapperV2::extract(const cv::Mat& image,
                                         const std::vector<cv::KeyPoint>& keypoints,
                                         const DescriptorParams& params) {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;

        wrapper_->computeDSP(image, kps, descriptors, params);
        last_descriptor_size_ = descriptors.empty() ? 0 : descriptors.cols;

        return descriptors;
    }

}
