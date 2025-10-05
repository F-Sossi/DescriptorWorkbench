// DNNPatchWrapper.hpp
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>
#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv2/dnn.hpp>

namespace thesis_project::wrappers {

    // Use the proper DescriptorParams from types.hpp
    using thesis_project::DescriptorParams;

    class DNNPatchWrapper : public IDescriptorExtractor {
    public:
        // Primary constructor
        DNNPatchWrapper(const std::string& onnx_model_path,
                        int input_size = 32,
                        float support_multiplier = 12.0f,
                        bool rotate_to_upright = true,
                        float mean = 0.0f,
                        float std = 1.0f,
                        bool per_patch_standardize = true,
                        int descriptor_size = 128);

        // Optionally specify explicit ONNX I/O names (recommended if your model uses them)
        void setInputOutputNames(const std::string& input_name, const std::string& output_name);

        // Prefer explicit backend/target once you're ready (CPU is the safest default)
        void setBackendTarget(int backend, int target);

        // Main API: extract descriptors for keypoints in 'image'
        cv::Mat extract(const cv::Mat& imageBgrOrGray,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params /* not used, kept for API compatibility */) override;

        // IDescriptorExtractor interface
        std::string name() const override { return "dnn_patch"; }
        int descriptorSize() const override { return descriptor_size_; }
        int descriptorType() const override { return CV_32F; }

    private:
        // Single-patch maker (grayscale + warp + float + normalization)
        cv::Mat makePatch_(const cv::Mat& imageGray, const cv::KeyPoint& kp) const;

    private:
        cv::dnn::Net net_;

        int   input_size_             = 32;    // N (e.g., 32)
        float support_mult_           = 12.0f; // support window relative to kp.size
        bool  rotate_upright_         = true;  // rotate patch to upright (undo kp.angle)

        float mean_                   = 0.0f;  // global mean (used if per_patch_standardize_ == false)
        float std_                    = 1.0f;  // global std
        bool  per_patch_standardize_  = true;  // z-score each patch individually

        int   descriptor_size_        = 128;   // expected C dimension

        // Optional (but helpful for clarity / debugging)
        std::string input_name_;               // ONNX input tensor name (empty = default)
        std::string output_name_;              // ONNX output tensor name (empty = default)

        // Tuning knob for batching
        int   default_batch_size_     = 512;
    };

}


/*
#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <opencv2/dnn.hpp>

namespace thesis_project {
namespace wrappers {

class DNNPatchWrapper : public IDescriptorExtractor {
private:
    cv::dnn::Net net_;
    int input_size_ = 32;
    float support_mult_ = 1.0f;
    bool rotate_upright_ = true;
    float mean_ = 0.0f;
    float std_ = 1.0f;
    bool per_patch_standardize_ = false;

public:
    explicit DNNPatchWrapper(const std::string& onnx_model_path,
                             int input_size,
                             float support_multiplier,
                             bool rotate_to_upright,
                             float mean,
                             float std,
                             bool per_patch_standardize);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params = {}) override;

    std::string name() const override { return "DNNPatch"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_SIFT; }
};

} // namespace wrappers
} // namespace thesis_project
*/
