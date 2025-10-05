#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "keypoints/RGBSIFT.h"

namespace thesis_project {
namespace wrappers {

class RGBSIFTWrapper : public IDescriptorExtractor {
private:
    std::unique_ptr<RGBSIFT> rgbsift_;

public:
    RGBSIFTWrapper();

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params = {}) override;

    std::string name() const override { return "RGBSIFT"; }
    int descriptorSize() const override { return 384; } // 3 * 128
    int descriptorType() const override { return static_cast<int>(thesis_project::DescriptorType::RGBSIFT); }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project
