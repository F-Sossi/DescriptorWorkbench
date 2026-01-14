#include "PatchCNNExtractor.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>


namespace thesis_project::patches {

#ifdef BUILD_LIBTORCH_DESCRIPTORS

PatchCNNExtractor::PatchCNNExtractor(const std::string& model_path,
                                     const std::string& name_str,
                                     int input_size,
                                     int descriptor_size,
                                     bool per_patch_standardize)
    : device_(torch::kCPU),
      model_path_(model_path),
      name_(name_str),
      input_size_(input_size),
      descriptor_size_(descriptor_size),
      per_patch_standardize_(per_patch_standardize) {

    try {
        // Load the TorchScript model
        model_ = torch::jit::load(model_path);

        // Auto-detect device (GPU if available)
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "PatchCNNExtractor [" << name_ << "]: Using GPU (CUDA)" << std::endl;
        } else {
            std::cout << "PatchCNNExtractor [" << name_ << "]: Using CPU" << std::endl;
        }

        // Move model to device and set to evaluation mode
        model_.to(device_);
        model_.eval();

        std::cout << "PatchCNNExtractor [" << name_ << "]: Loaded model from " << model_path << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error("PatchCNNExtractor: Failed to load model: " + std::string(e.what()));
    }
}

cv::Mat PatchCNNExtractor::extractFromPatches(
    const std::vector<cv::Mat>& patches,
    const DescriptorParams& params) {

    if (patches.empty()) {
        return cv::Mat();
    }

    // Determine target device from params
    torch::Device target_device = device_;
    if (params.device == "cpu") {
        target_device = torch::Device(torch::kCPU);
    } else if ((params.device == "cuda" || params.device == "auto") && torch::cuda::is_available()) {
        target_device = torch::Device(torch::kCUDA, 0);
    }

    // Move model if device changed
    const bool devices_different = (target_device.type() != device_.type());
    if (devices_different) {
        model_.to(target_device);
        device_ = target_device;
    }

    // Prepare batch of tensors
    std::vector<torch::Tensor> tensor_patches;
    tensor_patches.reserve(patches.size());

    for (const auto& patch : patches) {
        // Resize if necessary (65x65 -> 32x32)
        cv::Mat patch32;
        if (patch.cols != input_size_ || patch.rows != input_size_) {
            cv::resize(patch, patch32, cv::Size(input_size_, input_size_),
                      0, 0, cv::INTER_AREA);
        } else {
            patch32 = patch;
        }

        // Ensure grayscale
        cv::Mat patch_gray;
        if (patch32.channels() == 3) {
            cv::cvtColor(patch32, patch_gray, cv::COLOR_BGR2GRAY);
        } else {
            patch_gray = patch32;
        }

        tensor_patches.push_back(patchToTensor(patch_gray));
    }

    // Stack into batch tensor [N, 1, H, W]
    torch::Tensor batch = torch::stack(tensor_patches, 0).to(target_device);

    // Run inference
    cv::Mat descriptors;
    {
        torch::NoGradGuard no_grad;

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(batch);

        torch::Tensor output = model_.forward(inputs).toTensor();

        // Move back to CPU for OpenCV conversion
        output = output.to(torch::kCPU);

        descriptors = tensorToMat(output);
    }

    return descriptors;
}

std::unique_ptr<IPatchDescriptorExtractor> PatchCNNExtractor::clone() const {
    return std::make_unique<PatchCNNExtractor>(
        model_path_,
        name_,
        input_size_,
        descriptor_size_,
        per_patch_standardize_
    );
}

torch::Tensor PatchCNNExtractor::patchToTensor(const cv::Mat& patch32) const {
    // Convert to float and normalize to [0, 1]
    cv::Mat patch_float;
    patch32.convertTo(patch_float, CV_32F, 1.0 / 255.0);

    // Apply normalization
    if (per_patch_standardize_) {
        // Per-patch z-score normalization
        cv::Scalar mean, stddev;
        cv::meanStdDev(patch_float, mean, stddev);
        patch_float = (patch_float - mean[0]) / (stddev[0] + 1e-8);
    } else {
        // ImageNet-style normalization
        constexpr float imagenet_mean = 0.485f;
        constexpr float imagenet_std = 0.229f;
        patch_float = (patch_float - imagenet_mean) / imagenet_std;
    }

    // Convert to PyTorch tensor [1, H, W]
    torch::Tensor tensor = torch::from_blob(
        patch_float.data,
        {1, input_size_, input_size_},
        torch::kFloat32
    ).clone();  // Clone to ensure memory safety

    return tensor;
}

cv::Mat PatchCNNExtractor::tensorToMat(const torch::Tensor& tensor) {
    // Expected shape: [N, descriptor_size]
    const auto sizes = tensor.sizes();
    if (sizes.size() != 2) {
        throw std::runtime_error("Expected 2D tensor [N, D], got " +
                                std::to_string(sizes.size()) + "D");
    }

    int num_descriptors = static_cast<int>(sizes[0]);
    int desc_size = static_cast<int>(sizes[1]);

    // Ensure contiguous
    torch::Tensor cpu_tensor = tensor.contiguous();

    // Create OpenCV Mat
    cv::Mat descriptors(num_descriptors, desc_size, CV_32F);

    // Copy data
    std::memcpy(descriptors.data, cpu_tensor.data_ptr<float>(),
                num_descriptors * desc_size * sizeof(float));

    // L2 normalize each descriptor
    for (int i = 0; i < num_descriptors; ++i) {
        cv::Mat desc_row = descriptors.row(i);
        cv::normalize(desc_row, desc_row, 1.0, 0.0, cv::NORM_L2);
    }

    return descriptors;
}

// Factory functions
std::unique_ptr<IPatchDescriptorExtractor> createPatchHardNet() {
    return std::make_unique<PatchCNNExtractor>(
        "../models/hardnet.pt",  // Model path
        "hardnet",               // Name
        32,                      // Input size
        128,                     // Descriptor size
        true                     // Per-patch standardize
    );
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchSOSNet() {
    return std::make_unique<PatchCNNExtractor>(
        "../models/sosnet.pt",
        "sosnet",
        32,
        128,
        true
    );
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchL2Net() {
    return std::make_unique<PatchCNNExtractor>(
        "../models/simple_l2net.pt",
        "l2net",
        32,
        128,
        true
    );
}

#else
// Stub implementations when LibTorch is not available

PatchCNNExtractor::PatchCNNExtractor(const std::string& model_path,
                                     const std::string& name_str,
                                     int input_size,
                                     int descriptor_size,
                                     bool per_patch_standardize)
    : model_path_(model_path),
      name_(name_str),
      input_size_(input_size),
      descriptor_size_(descriptor_size),
      per_patch_standardize_(per_patch_standardize) {
    throw std::runtime_error("PatchCNNExtractor: LibTorch support not compiled. "
                            "Rebuild with BUILD_LIBTORCH_DESCRIPTORS=ON");
}

cv::Mat PatchCNNExtractor::extractFromPatches(
    const std::vector<cv::Mat>& /*patches*/,
    const DescriptorParams& /*params*/) {
    throw std::runtime_error("PatchCNNExtractor: LibTorch support not compiled");
}

std::unique_ptr<IPatchDescriptorExtractor> PatchCNNExtractor::clone() const {
    throw std::runtime_error("PatchCNNExtractor: LibTorch support not compiled");
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchHardNet() {
    throw std::runtime_error("HardNet requires LibTorch. Rebuild with BUILD_LIBTORCH_DESCRIPTORS=ON");
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchSOSNet() {
    throw std::runtime_error("SOSNet requires LibTorch. Rebuild with BUILD_LIBTORCH_DESCRIPTORS=ON");
}

std::unique_ptr<IPatchDescriptorExtractor> createPatchL2Net() {
    throw std::runtime_error("L2Net requires LibTorch. Rebuild with BUILD_LIBTORCH_DESCRIPTORS=ON");
}

#endif // BUILD_LIBTORCH_DESCRIPTORS

} // namespace thesis_project::patches

