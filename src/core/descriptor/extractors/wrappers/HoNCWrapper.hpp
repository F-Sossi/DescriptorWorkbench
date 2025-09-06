#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/HoNC.h"
#include <memory>

namespace thesis_project {
namespace wrappers {

class HoNCWrapper : public IDescriptorExtractor {
private:
    std::unique_ptr<HoNC> honc_;
    std::unique_ptr<experiment_config> config_;

public:
    HoNCWrapper();
    explicit HoNCWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params = {}) override;

    std::string name() const override { return "HoNC"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_HoNC; }

    std::string getConfiguration() const;
};

} // namespace wrappers
} // namespace thesis_project

