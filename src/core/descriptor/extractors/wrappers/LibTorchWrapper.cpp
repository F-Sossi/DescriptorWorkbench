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
                                 int descriptor_size)
    : device_(torch::kCPU),
      input_size_(input_size),
      support_mult_(support_multiplier),
      rotate_upright_(rotate_to_upright),
      mean_(mean),
      std_(std),
      per_patch_standardize_(per_patch_standardize),
      descriptor_size_(descriptor_size) {

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
    torch::Tensor batch = torch::stack(patches, 0).to(device_);

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
    // Calculate patch size based on keypoint size and support multiplier
    float patch_size = kp.size * support_mult_;
    if (patch_size <= 0) {
        patch_size = input_size_;
    }

    // Extract oriented patch
    cv::Mat patch;
    cv::Point2f center(kp.pt.x, kp.pt.y);

    if (rotate_upright_ && kp.angle != -1) {
        // Create rotation matrix to undo keypoint angle (make it upright)
        cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, -kp.angle, patch_size / input_size_);

        // Extract patch with rotation and scaling
        cv::Mat rotated;
        cv::warpAffine(imageGray, rotated, rotation_matrix, cv::Size(input_size_, input_size_));
        patch = rotated;
    } else {
        // Simple centered extraction without rotation
        float half_size = patch_size * 0.5f;
        cv::Rect roi(
            static_cast<int>(center.x - half_size),
            static_cast<int>(center.y - half_size),
            static_cast<int>(patch_size),
            static_cast<int>(patch_size)
        );

        // Ensure ROI is within image bounds
        roi &= cv::Rect(0, 0, imageGray.cols, imageGray.rows);

        if (roi.width > 0 && roi.height > 0) {
            cv::Mat cropped = imageGray(roi);
            cv::resize(cropped, patch, cv::Size(input_size_, input_size_));
        } else {
            // Fallback: create empty patch
            patch = cv::Mat::zeros(input_size_, input_size_, CV_8UC1);
        }
    }

    // Convert to tensor
    return matToTensor(patch);
}

torch::Tensor LibTorchWrapper::matToTensor(const cv::Mat& patch) const {
    // Convert to float and normalize to [0, 1]
    cv::Mat patch_float;
    patch.convertTo(patch_float, CV_32F, 1.0/255.0);

    // Apply normalization
    if (per_patch_standardize_) {
        // Per-patch standardization (z-score)
        cv::Scalar mean, stddev;
        cv::meanStdDev(patch_float, mean, stddev);
        patch_float = (patch_float - mean[0]) / (stddev[0] + 1e-8);
    } else {
        // Global normalization
        patch_float = (patch_float - mean_) / std_;
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