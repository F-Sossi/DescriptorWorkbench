// DNNPatchWrapper.cpp
#include "DNNPatchWrapper.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace thesis_project {
namespace wrappers {

DNNPatchWrapper::DNNPatchWrapper(const std::string& onnx_model_path,
                                 int input_size,
                                 float support_multiplier,
                                 bool rotate_to_upright,
                                 float mean,
                                 float std,
                                 bool per_patch_standardize,
                                 int descriptor_size)
    : input_size_(input_size),
      support_mult_(support_multiplier),
      rotate_upright_(rotate_to_upright),
      mean_(mean),
      std_(std),
      per_patch_standardize_(per_patch_standardize),
      descriptor_size_(descriptor_size) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnx_model_path);
        if (net_.empty()) {
            throw std::runtime_error("readNetFromONNX returned empty network");
        }
        // Default to safe, portable path first.
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("ONNX model loading failed: ") + e.what());
    }
}

void DNNPatchWrapper::setInputOutputNames(const std::string& input_name,
                                          const std::string& output_name) {
    input_name_ = input_name;
    output_name_ = output_name;
}

void DNNPatchWrapper::setBackendTarget(int backend, int target) {
    net_.setPreferableBackend(backend);
    net_.setPreferableTarget(target);
}

cv::Mat DNNPatchWrapper::makePatch_(const cv::Mat& imageGray, const cv::KeyPoint& kp) const {
    // kp.angle == -1 means "unset" -> treat as 0
    const float angle_deg = (rotate_upright_ && kp.angle >= 0.f) ? -kp.angle : 0.f;

    // Guard very small kp.size to avoid div-by-zero / insane scales
    const float kpSize = std::max(kp.size, 1.0f);
    const float S = std::max(1.0f, support_mult_ * kpSize);
    const double scale = static_cast<double>(input_size_) / static_cast<double>(S);

    // Build warp: rotate about kp.pt, scale, then translate to center the kp at (N/2,N/2)
    cv::Mat M = cv::getRotationMatrix2D(kp.pt, angle_deg, scale);
    M.at<double>(0, 2) += (input_size_ * 0.5 - kp.pt.x);
    M.at<double>(1, 2) += (input_size_ * 0.5 - kp.pt.y);

    cv::Mat patch8u;
    cv::warpAffine(imageGray, patch8u, M, cv::Size(input_size_, input_size_),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat patch32f;
    patch8u.convertTo(patch32f, CV_32F, 1.0 / 255.0);  // [0,1]

    if (per_patch_standardize_) {
        cv::Scalar mu, sigma;
        cv::meanStdDev(patch32f, mu, sigma);
        const float eps = 1e-6f;
        const float denom = static_cast<float>(std::max(sigma[0], (double)eps));
        patch32f = (patch32f - static_cast<float>(mu[0])) / denom;
    } else if (mean_ != 0.0f || std_ != 1.0f) {
        patch32f = (patch32f - mean_) / std_;
    }

    return patch32f; // single-channel float N×N
}

cv::Mat DNNPatchWrapper::extract(const cv::Mat& imageBgrOrGray,
                                 const std::vector<cv::KeyPoint>& keypoints,
                                 const DescriptorParams& /*params*/) {
    // ---- 0) Ensure single-channel source once
    cv::Mat imageGray;
    if (imageBgrOrGray.channels() == 1) {
        imageGray = imageBgrOrGray;
    } else {
        cv::cvtColor(imageBgrOrGray, imageGray, cv::COLOR_BGR2GRAY);
    }

    const int N = input_size_;
    const int C_expected = descriptor_size_;
    const int totalKp = static_cast<int>(keypoints.size());
    cv::Mat descriptors(totalKp, C_expected, CV_32F, cv::Scalar(0));

    if (totalKp == 0) return descriptors;

    // Batching
    const int BATCH = std::max(1, default_batch_size_);
    std::vector<cv::Mat> batch;
    batch.reserve(std::min(totalKp, BATCH));

    int start = 0;
    while (start < totalKp) {
        batch.clear();
        const int end = std::min(start + BATCH, totalKp);

        for (int i = start; i < end; ++i) {
            batch.push_back(makePatch_(imageGray, keypoints[i]));
        }

        // blobFromImages -> shape (B,1,N,N)
        cv::Mat blob = cv::dnn::blobFromImages(
            batch, /*scalefactor=*/1.0, cv::Size(N, N),
            cv::Scalar(), /*swapRB=*/false, /*crop=*/false, CV_32F);

        try {
            if (!input_name_.empty())
                net_.setInput(blob, input_name_);
            else
                net_.setInput(blob);

            cv::Mat out = output_name_.empty() ? net_.forward() : net_.forward(output_name_);

            // Normalize output to shape [B, C]
            const int B = end - start;
            cv::Mat out2;
            if (out.dims == 4 && out.size[2] == 1 && out.size[3] == 1) {
                // [B, C, 1, 1] -> [B, C]
                out2 = out.reshape(1, out.size[0]);
            } else if (out.dims == 4 && (out.size[2] > 1 || out.size[3] > 1)) {
                // [B, C, H, W] -> global average pool to [B, C]
                int Bn = out.size[0], Cn = out.size[1], H = out.size[2], W = out.size[3];
                if (out.type() != CV_32F) {
                    cv::Mat tmp;
                    out.convertTo(tmp, CV_32F);
                    out = tmp;
                }
                out2.create(Bn, Cn, CV_32F);
                const int HW = H * W;
                for (int b = 0; b < Bn; ++b) {
                    for (int c = 0; c < Cn; ++c) {
                        const float* p = out.ptr<float>(b, c);
                        double sum = 0.0;
                        for (int i = 0; i < HW; ++i) sum += p[i];
                        out2.at<float>(b, c) = static_cast<float>(sum / (double)HW);
                    }
                }
            } else if (out.dims == 2) {
                // [B, C]
                out2 = out;
            } else {
                // Fallback: flatten to [B, C]
                int total = static_cast<int>(out.total());
                if (B <= 0 || total % B != 0) {
                    throw std::runtime_error("Unexpected DNN output shape (not divisible by batch).");
                }
                int C = total / B;
                out2 = out.reshape(1, B);
                if (out2.cols != C) {
                    out2 = out.reshape(1, B); // ensure rows=B, cols computed
                }
            }

            if (out2.type() != CV_32F) out2.convertTo(out2, CV_32F);

            // Handle two cases: [B, C_raw] or [1, B*C_raw]
            if (out2.rows == B) {
                const int C = out2.cols;
                for (int b = 0; b < B; ++b) {
                    cv::Mat srcRow = out2.row(b);
                    cv::Mat dstRow = descriptors.row(start + b);
                    if (C >= C_expected) {
                        srcRow.colRange(0, C_expected).copyTo(dstRow);
                    } else {
                        if (C > 0) srcRow.colRange(0, C).copyTo(dstRow.colRange(0, C));
                        if (C < C_expected) dstRow.colRange(C, C_expected).setTo(0);
                    }
                    cv::normalize(dstRow, dstRow, 1.0, 0.0, cv::NORM_L2);
                }
            } else if (out2.rows == 1 && out2.cols % B == 0) {
                const int C_raw = out2.cols / B;
                for (int b = 0; b < B; ++b) {
                    int startCol = b * C_raw;
                    int endCol = (b + 1) * C_raw;
                    cv::Mat srcSeg = out2.colRange(startCol, endCol);
                    cv::Mat dstRow = descriptors.row(start + b);
                    if (C_raw >= C_expected) {
                        srcSeg.colRange(0, C_expected).copyTo(dstRow);
                    } else {
                        if (C_raw > 0) srcSeg.colRange(0, C_raw).copyTo(dstRow.colRange(0, C_raw));
                        if (C_raw < C_expected) dstRow.colRange(C_raw, C_expected).setTo(0);
                    }
                    cv::normalize(dstRow, dstRow, 1.0, 0.0, cv::NORM_L2);
                }
            } else {
                throw std::runtime_error("Unexpected DNN output layout after forward");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("DNN forward pass failed: ") + e.what());
        }

        start = end;
    }

    return descriptors;
}

} // namespace wrappers
} // namespace thesis_project



/*
#include "DNNPatchWrapper.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace thesis_project {
namespace wrappers {

DNNPatchWrapper::DNNPatchWrapper(const std::string& onnx_model_path,
                                 int input_size,
                                 float support_multiplier,
                                 bool rotate_to_upright,
                                 float mean,
                                 float std,
                                 bool per_patch_standardize)
    : input_size_(input_size),
      support_mult_(support_multiplier),
      rotate_upright_(rotate_to_upright),
      mean_(mean),
      std_(std),
      per_patch_standardize_(per_patch_standardize)
{
    try {
        net_ = cv::dnn::readNetFromONNX(onnx_model_path);
        if (net_.empty()) {
            throw std::runtime_error("Failed to load ONNX model - network is empty");
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("ONNX model loading failed: " + std::string(e.what()));
    }
}

static cv::Mat extractPatch(const cv::Mat& image, const cv::KeyPoint& kp, int N, float support_mult, bool rotate_upright) {
    const float S = std::max(1.0f, support_mult * kp.size);
    // To rotate the patch to upright, we must undo the keypoint angle
    const float angle_deg = rotate_upright ? -kp.angle : 0.0f;
    const double scale = static_cast<double>(N) / static_cast<double>(S);
    // Build rotation (about kp.pt) and scale, then translate so kp maps to patch center
    cv::Mat M = cv::getRotationMatrix2D(kp.pt, angle_deg, scale); // 2x3
    // Shift so that kp.pt -> (N/2, N/2)
    M.at<double>(0, 2) += (static_cast<double>(N) * 0.5 - kp.pt.x);
    M.at<double>(1, 2) += (static_cast<double>(N) * 0.5 - kp.pt.y);
    cv::Mat patch;
    cv::warpAffine(image, patch, M, cv::Size(N, N), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return patch;
}

cv::Mat DNNPatchWrapper::extract(const cv::Mat& image,
                                 const std::vector<cv::KeyPoint>& keypoints,
                                 const DescriptorParams& params) {
    const int N = input_size_;
    cv::Mat descriptors(static_cast<int>(keypoints.size()), descriptorSize(), CV_32F, cv::Scalar(0));
    int row = 0;
    for (const auto& kp : keypoints) {
        cv::Mat patch = extractPatch(image, kp, N, support_mult_, rotate_upright_);
        cv::Mat patchF;
        patch.convertTo(patchF, CV_32F, 1.0/255.0);
        // Optional per-patch standardization to zero mean / unit variance
        if (per_patch_standardize_) {
            cv::Scalar mu, sigma;
            cv::meanStdDev(patchF, mu, sigma);
            float eps = 1e-6f;
            float denom = static_cast<float>(sigma[0]) > eps ? static_cast<float>(sigma[0]) : eps;
            patchF = (patchF - static_cast<float>(mu[0])) / denom;
        } else {
            if (mean_ != 0.0f || std_ != 1.0f) patchF = (patchF - mean_) / std_;
        }
        cv::Mat blob = cv::dnn::blobFromImage(patchF); // CHW
        net_.setInput(blob);
        cv::Mat out;
        try {
            out = net_.forward();
            if (out.empty()) {
                throw std::runtime_error("DNN forward pass returned empty Mat");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("DNN forward pass failed: " + std::string(e.what()));
        }
        
        // Handle different output shapes (1xC or 1xCx1x1)
        if (out.dims == 4 && out.size[2] == 1 && out.size[3] == 1) {
            // Reshape from [1,C,1,1] to [1,C]
            out = out.reshape(1, out.size[1]);
        } else if (out.dims == 2) {
            // Already [1,C], no reshape needed
        } else {
            // Fallback: flatten to 1D and take as descriptor
            out = out.reshape(1, 1);
        }
        if (out.type() != CV_32F) {
            out.convertTo(out, CV_32F);
        }
        // L2 normalise
        cv::Mat dstRow = descriptors.row(row);
        if (out.cols >= dstRow.cols) {
            out.colRange(0, dstRow.cols).copyTo(dstRow);
        } else {
            dstRow.colRange(0, out.cols) = out;
        }
        cv::normalize(dstRow, dstRow, 1.0, 0.0, cv::NORM_L2);
        row++;
    }
    return descriptors;
}

} // namespace wrappers
} // namespace thesis_project
*/
