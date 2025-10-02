#include "RGBSIFTWrapper.hpp"
#include <sstream>

namespace thesis_project {
namespace wrappers {

RGBSIFTWrapper::RGBSIFTWrapper() {
    rgbsift_ = std::make_unique<RGBSIFT>();
}

RGBSIFTWrapper::RGBSIFTWrapper(const experiment_config& config)
    : config_(std::make_unique<experiment_config>(config)) {
    rgbsift_ = std::make_unique<RGBSIFT>();
}

cv::Mat RGBSIFTWrapper::extract(const cv::Mat& image,
                               const std::vector<cv::KeyPoint>& keypoints,
                               const DescriptorParams& params) {
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> mutable_keypoints = keypoints;

    // RGBSIFT inherits from DSPSIFT, so compute() uses DSP internally
    rgbsift_->compute(image, mutable_keypoints, descriptors);

    // Apply RootSIFT transformation if requested
    if (params.rooting_stage == RootingStage::R_AFTER_POOLING) {
        // L1 normalize then sqrt (RootSIFT)
        for (int r = 0; r < descriptors.rows; r++) {
            cv::Mat row = descriptors.row(r);
            cv::normalize(row, row, 1.0, 0.0, cv::NORM_L1);
        }
        // Apply element-wise sqrt
        for (int r = 0; r < descriptors.rows; r++) {
            float* ptr = descriptors.ptr<float>(r);
            for (int c = 0; c < descriptors.cols; c++) {
                float v = ptr[c];
                ptr[c] = (v < 0.0f) ? 0.0f : std::sqrt(v);
            }
        }
    }

    // Final normalization
    if (params.normalize_after_pooling) {
        for (int r = 0; r < descriptors.rows; r++) {
            cv::Mat row = descriptors.row(r);
            cv::normalize(row, row, 1.0, 0.0, params.norm_type);
        }
    }

    return descriptors;
}

std::string RGBSIFTWrapper::getConfiguration() const {
    std::stringstream ss;
    ss << "RGBSIFT Wrapper Configuration:\n";
    ss << "  RGB SIFT descriptor\n";
    ss << "  Descriptor size: " << descriptorSize() << "\n";
    if (config_) {
        ss << "  Pooling Strategy: " << static_cast<int>(config_->descriptorOptions.poolingStrategy) << "\n";
    }
    return ss.str();
}

} // namespace wrappers
} // namespace thesis_project
