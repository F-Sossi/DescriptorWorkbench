#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv4/opencv2/features2d.hpp>
#include "src/core/config/experiment_config.hpp"


namespace thesis_project::wrappers {

class SIFTWrapper final : public IDescriptorExtractor {
private:
    cv::Ptr<cv::SIFT> sift_;
    std::unique_ptr<experiment_config> config_;

public:
    SIFTWrapper();
    explicit SIFTWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override;

    [[nodiscard]] std::string name() const override { return "SIFT"; }
    [[nodiscard]] int descriptorSize() const override { return 128; }
    [[nodiscard]] int descriptorType() const override { return DESCRIPTOR_SIFT; }

    [[nodiscard]] std::string getConfiguration() const;
};

} // namespace thesis_project::wrappers

