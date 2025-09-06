#pragma once

#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
#include <memory>

// Guard include of xfeatures2d for portability
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif

namespace thesis_project {
namespace wrappers {

class VGGWrapper : public IDescriptorExtractor {
public:
    VGGWrapper();
    explicit VGGWrapper(const experiment_config& config);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const thesis_project::DescriptorParams& params) override;

    std::string name() const override { return "VGG"; }

    int descriptorSize() const override {
        if (extractor_) return extractor_->descriptorSize();
        // Default VGG descriptor size is typically 120 or 128 depending on settings
        return 120;
    }

    int descriptorType() const override { return CV_32F; }

private:
    cv::Ptr<cv::Feature2D> extractor_;
    std::unique_ptr<experiment_config> config_;
};

} // namespace wrappers
} // namespace thesis_project

