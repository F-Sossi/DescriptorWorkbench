#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv2/features2d.hpp>
#include "src/core/config/experiment_config.hpp"

namespace thesis_project {
namespace wrappers {

class SIFTWrapper : public IDescriptorExtractor {
private:
    cv::Ptr<cv::SIFT> sift_;
    std::unique_ptr<experiment_config> config_;

public:
    SIFTWrapper();
    explicit SIFTWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params = {}) override;

    std::string name() const override { return "SIFT"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_SIFT; }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project
