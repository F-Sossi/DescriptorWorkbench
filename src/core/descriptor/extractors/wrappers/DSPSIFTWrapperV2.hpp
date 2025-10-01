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
    DSPSIFTWrapperV2() {
        wrapper_ = cv::makePtr<DSPVanillaSIFTWrapper<VanillaSIFT>>();
    }

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;  // Non-const copy

        // If no scales provided, use default
        std::vector<float> scales = params.scales.empty()
            ? std::vector<float>{0.85f, 1.0f, 1.30f}
            : params.scales;

        // Use the pyramid-aware DSP compute
        wrapper_->computeDSP(image, kps, descriptors, scales, params.pooling_aggregation);

        return descriptors;
    }

    int descriptorSize() const override {
        // Base SIFT is 128, but concatenate mode increases it
        return 128;  // Will be 384 for concatenate
    }

    int descriptorType() const override {
        return CV_32F;
    }

    std::string name() const override {
        return "DSPSIFTWrapperV2";
    }

private:
    cv::Ptr<DSPVanillaSIFTWrapper<VanillaSIFT>> wrapper_;
};

} // namespace thesis_project
