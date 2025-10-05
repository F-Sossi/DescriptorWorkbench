#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv4/opencv2/features2d.hpp>

namespace thesis_project::wrappers {

class ORBWrapper final : public IDescriptorExtractor {
private:
    cv::Ptr<cv::ORB> orb_;

public:
    ORBWrapper();

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override;

    [[nodiscard]] std::string name() const override { return "ORB"; }
    [[nodiscard]] int descriptorSize() const override { return 32; }  // ORB uses 32-byte descriptors
    [[nodiscard]] int descriptorType() const override { return CV_8U; } // ORB uses binary descriptors

    [[nodiscard]] std::string getConfiguration() const;
};

} // namespace thesis_project::wrappers
