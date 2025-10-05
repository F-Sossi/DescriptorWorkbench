#include "DSPHoWHWrapperV2.hpp"

namespace thesis_project::wrappers {

    DSPHoWHWrapperV2::DSPHoWHWrapperV2()
        : wrapper_(cv::makePtr<DSPVanillaSIFTWrapper<HoWH>>()) {}

    cv::Mat DSPHoWHWrapperV2::extract(const cv::Mat& image,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      const DescriptorParams& params) {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;

        wrapper_->computeDSP(image, kps, descriptors, params);
        last_descriptor_size_ = descriptors.empty() ? 0 : descriptors.cols;

        return descriptors;
    }

}
