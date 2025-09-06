#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/VanillaSIFT.h"
#include <memory>

namespace thesis_project {
namespace wrappers {

class VSIFTWrapper : public IDescriptorExtractor {
private:
    cv::Ptr<VanillaSIFT> vsift_;
    std::unique_ptr<experiment_config> config_;

public:
    VSIFTWrapper();
    explicit VSIFTWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params = {}) override;

    std::string name() const override { return "vSIFT"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_vSIFT; }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project

