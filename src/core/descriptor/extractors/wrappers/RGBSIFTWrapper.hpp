#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "keypoints/RGBSIFT.h"
#include "src/core/config/experiment_config.hpp"

namespace thesis_project {
namespace wrappers {

class RGBSIFTWrapper : public IDescriptorExtractor {
private:
    std::unique_ptr<RGBSIFT> rgbsift_;
    std::unique_ptr<experiment_config> config_;

public:
    RGBSIFTWrapper();
    explicit RGBSIFTWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params = {}) override;

    std::string name() const override { return "RGBSIFT"; }
    int descriptorSize() const override { return 384; } // 3 * 128
    int descriptorType() const override { return DESCRIPTOR_RGBSIFT; }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project
