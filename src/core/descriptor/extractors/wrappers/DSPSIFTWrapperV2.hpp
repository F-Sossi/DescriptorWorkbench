#pragma once

#include "src/interfaces/IDescriptorExtractor.hpp"
#include "keypoints/DSPVanillaSIFTWrapper.h"
#include "include/thesis_project/types.hpp"
#include <opencv2/opencv.hpp>

namespace thesis_project {

/**
 * @brief IDescriptorExtractor adapter for DSPVanillaSIFTWrapper
 *
 * Wraps the DSPVanillaSIFTWrapper to work with the experiment pipeline.
 * This provides true pyramid-aware DSP that should match DSPSIFT performance.
 */
class DSPSIFTWrapperV2 : public IDescriptorExtractor {
public:
    DSPSIFTWrapperV2()
        : wrapper_(cv::makePtr<DSPVanillaSIFTWrapper<VanillaSIFT>>()) {}

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;  // Non-const copy

        wrapper_->computeDSP(image, kps, descriptors, params);

        last_descriptor_size_ = descriptors.empty() ? 0 : descriptors.cols;

        return descriptors;
    }

    int descriptorSize() const override {
        return last_descriptor_size_;
    }

    int descriptorType() const override {
        return CV_32F;
    }

    std::string name() const override {
        return "DSPSIFTWrapperV2";
    }

private:
    cv::Ptr<DSPVanillaSIFTWrapper<VanillaSIFT>> wrapper_;
    mutable int last_descriptor_size_ = 128;
};

} // namespace thesis_project
