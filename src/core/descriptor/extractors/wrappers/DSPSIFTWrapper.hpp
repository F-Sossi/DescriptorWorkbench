#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/DSPSIFT.h"
#include <memory>

namespace thesis_project {
namespace wrappers {

class DSPSIFTWrapper : public IDescriptorExtractor {
private:
    cv::Ptr<DSPSIFT> dspsift_;
    std::unique_ptr<experiment_config> config_;

public:
    DSPSIFTWrapper();
    explicit DSPSIFTWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params = {}) override;

    std::string name() const override { return "DSPSIFT"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_SIFT; }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project

