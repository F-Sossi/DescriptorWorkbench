// LibTorchWrapper.cpp
#include "LibTorchWrapper.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace thesis_project {
namespace wrappers {

LibTorchWrapper::LibTorchWrapper(const std::string& model_path,
                                 int input_size,
                                 float support_multiplier,
                                 bool rotate_to_upright,
                                 float mean,
                                 float std,
                                 bool per_patch_standardize,
                                 int descriptor_size,
                                 int resize_method)
    : device_(torch::kCPU),
      input_size_(input_size),
      support_mult_(support_multiplier),
      rotate_upright_(rotate_to_upright),
      mean_(mean),
      std_(std),
      per_patch_standardize_(per_patch_standardize),
      descriptor_size_(descriptor_size),
      resize_method_(resize_method) {

    try {
        // Load the model
        model_ = torch::jit::load(model_path);

        // Auto-detect device (GPU if available, CPU fallback)
        if (::torch::cuda::is_available()) {
            device_ = ::torch::Device(::torch::kCUDA, 0);
            std::cout << "LibTorch: Using GPU acceleration (CUDA)" << std::endl;
        } else {
            std::cout << "LibTorch: Using CPU" << std::endl;
        }

        // Move model to device and set to evaluation mode
        model_.to(device_);
        model_.eval();

        std::cout << "LibTorch: Successfully loaded model from " << model_path << std::endl;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("LibTorch model loading failed: ") + e.what());
    }
}

cv::Mat LibTorchWrapper::extract(const cv::Mat& imageBgrOrGray,
                                 const std::vector<cv::KeyPoint>& keypoints,
                                 const DescriptorParams& params) {
    // Debug: Show device settings
    std::cout << "LibTorch DEBUG: params.device = '" << params.device << "'" << std::endl;
    std::cout << "LibTorch DEBUG: constructor device_ = " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Override device if specified in params
    torch::Device target_device = device_;
    if (params.device == "cpu") {
        std::cout << "LibTorch DEBUG: Setting target device to CPU" << std::endl;
        target_device = torch::Device(torch::kCPU);
    } else if (params.device == "cuda" && torch::cuda::is_available()) {
        std::cout << "LibTorch DEBUG: Setting target device to CUDA" << std::endl;
        target_device = torch::Device(torch::kCUDA, 0);
    }
    // "auto" uses the device_ set in constructor

    std::cout << "LibTorch DEBUG: target_device = " << (target_device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Log device override if different from default
    bool devices_different = (target_device.type() != device_.type()) ||
                            (target_device.is_cuda() && device_.is_cuda() && target_device.index() != device_.index());

    if (devices_different) {
        std::cout << "LibTorch: Device overridden to " << (target_device.is_cuda() ? "CUDA" : "CPU")
                  << " via YAML config" << std::endl;
        // Move model to new device if different from construction device
        model_.to(target_device);
    } else {
        std::cout << "LibTorch DEBUG: No device change needed" << std::endl;
    }

    if (keypoints.empty()) {
        return cv::Mat();
    }

    // Convert to grayscale if needed
    cv::Mat imageGray;
    if (imageBgrOrGray.channels() == 3) {
        cv::cvtColor(imageBgrOrGray, imageGray, cv::COLOR_BGR2GRAY);
    } else {
        imageGray = imageBgrOrGray;
    }

    // Prepare batch tensor
    std::vector<torch::Tensor> patches;
    patches.reserve(keypoints.size());

    // Extract patches
    for (const auto& kp : keypoints) {
        torch::Tensor patch = makePatch(imageGray, kp);
        patches.push_back(patch);
    }

    // Stack patches into batch tensor [N, 1, H, W]
    torch::Tensor batch = torch::stack(patches, 0).to(target_device);

    // Run inference
    cv::Mat descriptors;
    {
        torch::NoGradGuard no_grad;

        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch);

        torch::Tensor output = model_.forward(inputs).toTensor();

        // Move back to CPU for OpenCV conversion
        output = output.to(torch::kCPU);

        // Convert to OpenCV Mat
        descriptors = tensorToMat(output);
    }

    return descriptors;
}

torch::Tensor LibTorchWrapper::makePatch(const cv::Mat& imageGray, const cv::KeyPoint& kp) const {
    // Direct 32×32 extraction (test reverting from HPatches protocol)
    const int CNN_INPUT_SIZE = input_size_;  // 32

    // Calculate scale using Kornia LAF protocol: kp.size is diameter, convert to radius
    float kpSize = std::max(kp.size, 1e-3f);  // Guard against zero/negative keypoint size
    float scale_radius = kpSize / 2.0f;  // Convert diameter to radius (Kornia standard)
    // Scale to map radius to half the patch size (32/2 = 16 pixels for radius)
    double scale = static_cast<double>(CNN_INPUT_SIZE) / (2.0 * static_cast<double>(scale_radius));

    // Determine rotation angle (undo keypoint angle to make patch upright)
    float angle_deg = (rotate_upright_ && kp.angle >= 0.0f) ? -kp.angle : 0.0f;

    // Build transformation matrix to match Kornia LAF protocol
    // First translate keypoint to origin, rotate, scale, then translate to patch center
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle_deg, scale);

    // Apply keypoint-centered transformation: translate kp to origin, then to patch center
    double cx = CNN_INPUT_SIZE * 0.5;  // Patch center x
    double cy = CNN_INPUT_SIZE * 0.5;  // Patch center y
    M.at<double>(0, 2) = cx - M.at<double>(0, 0) * kp.pt.x - M.at<double>(0, 1) * kp.pt.y;
    M.at<double>(1, 2) = cy - M.at<double>(1, 0) * kp.pt.x - M.at<double>(1, 1) * kp.pt.y;

    // Direct 32×32 extraction
    cv::Mat cnn_patch;
    cv::warpAffine(imageGray, cnn_patch, M, cv::Size(CNN_INPUT_SIZE, CNN_INPUT_SIZE),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // Convert to tensor
    return matToTensor(cnn_patch);
}

torch::Tensor LibTorchWrapper::matToTensor(const cv::Mat& patch) const {
    // Convert to float and normalize to [0, 1] first
    cv::Mat patch_float;
    patch.convertTo(patch_float, CV_32F, 1.0/255.0);

    // Apply HardNet-compatible normalization
    if (per_patch_standardize_) {
        // Per-patch standardization (z-score) - matches Kornia's approach
        cv::Scalar mean, stddev;
        cv::meanStdDev(patch_float, mean, stddev);
        patch_float = (patch_float - mean[0]) / (stddev[0] + 1e-8);
    } else {
        // Use ImageNet-style normalization (typical for HardNet training)
        // ImageNet single-channel approximation: mean=0.485, std=0.229
        const float imagenet_mean = 0.485f;
        const float imagenet_std = 0.229f;
        patch_float = (patch_float - imagenet_mean) / imagenet_std;
    }

    // Convert OpenCV Mat to PyTorch tensor
    // OpenCV: HxW -> PyTorch: 1xHxW (add channel dimension)
    torch::Tensor tensor = torch::from_blob(
        patch_float.data,
        {1, input_size_, input_size_},
        torch::kFloat32
    ).clone(); // Clone to ensure memory safety

    return tensor;
}

cv::Mat LibTorchWrapper::tensorToMat(const torch::Tensor& tensor) const {
    // Tensor shape should be [N, descriptor_size]
    auto sizes = tensor.sizes();
    if (sizes.size() != 2) {
        throw std::runtime_error("Expected 2D tensor [N, descriptor_size], got " +
                                 std::to_string(sizes.size()) + "D");
    }

    int num_descriptors = static_cast<int>(sizes[0]);
    int desc_size = static_cast<int>(sizes[1]);

    // Ensure tensor is contiguous and on CPU
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

} // namespace wrappers
} // namespace thesis_project