#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>

namespace thesis_project::wrappers {

class SURFWrapper final : public IDescriptorExtractor {
private:
    cv::Ptr<cv::xfeatures2d::SURF> surf_;
    bool extended_ = false; // track whether current instance is using extended descriptors

public:
    SURFWrapper();

    cv::Mat extract(const cv::Mat& image,
                   const std::vector<cv::KeyPoint>& keypoints,
                   const DescriptorParams& params) override;

    [[nodiscard]] std::string name() const override { return "SURF"; }
    [[nodiscard]] int descriptorSize() const override {
        // Query actual configuration from the underlying SURF instance
        if (surf_) return surf_->descriptorSize();
        return extended_ ? 128 : 64;
    }  // SURF uses 64-dimensional descriptors by default (128 when extended=true)
    [[nodiscard]] int descriptorType() const override { return CV_32F; } // SURF uses float descriptors

    [[nodiscard]] std::string getConfiguration() const;
};

} // namespace thesis_project::wrappers
