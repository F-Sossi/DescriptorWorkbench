// LibTorchWrapper.hpp
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"

namespace thesis_project {
namespace wrappers {

// Use the proper DescriptorParams from types.hpp
using thesis_project::DescriptorParams;

class LibTorchWrapper : public IDescriptorExtractor {
public:
    // Primary constructor
    LibTorchWrapper(const std::string& model_path,
                    int input_size = 32,
                    float support_multiplier = 5.0f,
                    bool rotate_to_upright = true,
                    float mean = 0.0f,
                    float std = 1.0f,
                    bool per_patch_standardize = true,
                    int descriptor_size = 128,
                    int resize_method = cv::INTER_AREA);

    // Main API: extract descriptors for keypoints in 'image'
    cv::Mat extract(const cv::Mat& imageBgrOrGray,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params /* not used, kept for API compatibility */) override;

    // IDescriptorExtractor interface
    std::string name() const override { return "libtorch_cnn"; }
    int descriptorSize() const override { return descriptor_size_; }
    int descriptorType() const override { return CV_32F; }

private:
    // Single-patch maker (grayscale + warp + tensor conversion)
    torch::Tensor makePatch(const cv::Mat& imageGray, const cv::KeyPoint& kp) const;

    // Convert OpenCV Mat to PyTorch tensor
    torch::Tensor matToTensor(const cv::Mat& patch) const;

    // Convert PyTorch tensor back to OpenCV Mat
    cv::Mat tensorToMat(const torch::Tensor& tensor) const;

private:
    torch::jit::script::Module model_;
    torch::Device device_;

    int   input_size_             = 32;    // N (e.g., 32)
    float support_mult_           = 6.0f;  // support window relative to kp.size
    bool  rotate_upright_         = true;  // rotate patch to upright (undo kp.angle)

    float mean_                   = 0.0f;  // global mean (used if per_patch_standardize_ == false)
    float std_                    = 1.0f;  // global std
    bool  per_patch_standardize_  = false; // z-score each patch individually

    int   descriptor_size_        = 128;   // expected output dimension
    int   resize_method_          = cv::INTER_AREA;  // OpenCV resize interpolation method

    // Tuning knob for batching
    int   default_batch_size_     = 512;
};

} // namespace wrappers
} // namespace thesis_project