#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include "src/core/config/experiment_config.hpp"

namespace thesis_project::wrappers {

class SURFWrapper final : public IDescriptorExtractor {
private:
    cv::Ptr<cv::xfeatures2d::SURF> surf_;
    std::unique_ptr<experiment_config> config_;

public:
    SURFWrapper();
    explicit SURFWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override;

    [[nodiscard]] std::string name() const override { return "SURF"; }
    [[nodiscard]] int descriptorSize() const override { return 64; }  // SURF uses 64-dimensional descriptors
    [[nodiscard]] int descriptorType() const override { return CV_32F; } // SURF uses float descriptors

    [[nodiscard]] std::string getConfiguration() const;
};

} // namespace thesis_project::wrappers