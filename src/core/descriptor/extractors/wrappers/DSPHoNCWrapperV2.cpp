#include "DSPHoNCWrapperV2.hpp"

namespace thesis_project::wrappers {

    DSPHoNCWrapperV2::DSPHoNCWrapperV2()
        : wrapper_(cv::makePtr<DSPVanillaSIFTWrapper<HoNC>>()) {}

    cv::Mat DSPHoNCWrapperV2::extract(const cv::Mat& image,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      const DescriptorParams& params) {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;

        wrapper_->computeDSP(image, kps, descriptors, params);
        last_descriptor_size_ = descriptors.empty() ? 0 : descriptors.cols;

        return descriptors;
    }

}
