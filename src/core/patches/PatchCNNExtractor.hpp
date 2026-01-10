#pragma once

#include "PatchDescriptorExtractor.hpp"
#include "PatchLoader.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifdef BUILD_LIBTORCH_DESCRIPTORS
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include <string>
#include <memory>

namespace thesis_project {
namespace patches {

/**
 * @brief Extracts CNN descriptors (HardNet, SOSNet) from pre-extracted patches
 *
 * This class handles:
 * 1. Resizing 65x65 patches to 32x32 (with INTER_AREA anti-aliasing)
 * 2. Per-patch z-score normalization
 * 3. Batched inference through LibTorch models
 * 4. L2 normalization of output descriptors
 */
class PatchCNNExtractor : public IPatchDescriptorExtractor {
public:
    /**
     * @brief Construct a CNN patch extractor
     * @param model_path Path to the TorchScript model (.pt file)
     * @param name_str Human-readable name (e.g., "hardnet", "sosnet")
     * @param input_size Expected model input size (default 32)
     * @param descriptor_size Output descriptor dimension (default 128)
     * @param per_patch_standardize Whether to apply per-patch z-score (default true)
     */
    explicit PatchCNNExtractor(const std::string& model_path,
                               const std::string& name_str = "cnn",
                               int input_size = 32,
                               int descriptor_size = 128,
                               bool per_patch_standardize = true);

    ~PatchCNNExtractor() override = default;

    /**
     * @brief Extract descriptors from pre-extracted patches
     * @param patches Vector of 65x65 grayscale patches
     * @param params Descriptor parameters (device selection, etc.)
     * @return cv::Mat where each row is a 128D L2-normalized descriptor
     */
    cv::Mat extractFromPatches(
        const std::vector<cv::Mat>& patches,
        const DescriptorParams& params) override;

    std::string name() const override { return name_; }
    int descriptorSize() const override { return descriptor_size_; }
    int descriptorType() const override { return CV_32F; }
    bool requiresResize() const override { return true; }
    int expectedPatchSize() const override { return input_size_; }
    std::unique_ptr<IPatchDescriptorExtractor> clone() const override;

private:
#ifdef BUILD_LIBTORCH_DESCRIPTORS
    /**
     * @brief Convert a 32x32 patch to a PyTorch tensor with normalization
     */
    torch::Tensor patchToTensor(const cv::Mat& patch32) const;

    /**
     * @brief Convert output tensor to OpenCV Mat with L2 normalization
     */
    static cv::Mat tensorToMat(const torch::Tensor& tensor);

    torch::jit::script::Module model_;
    torch::Device device_;
#endif

    std::string model_path_;
    std::string name_;
    int input_size_;
    int descriptor_size_;
    bool per_patch_standardize_;
    int batch_size_ = 512;
};

/**
 * @brief Factory functions for common CNN extractors
 */
std::unique_ptr<IPatchDescriptorExtractor> createPatchHardNet();
std::unique_ptr<IPatchDescriptorExtractor> createPatchSOSNet();
std::unique_ptr<IPatchDescriptorExtractor> createPatchL2Net();

} // namespace patches
} // namespace thesis_project
