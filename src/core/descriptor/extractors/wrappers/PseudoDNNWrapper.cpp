#include "PseudoDNNWrapper.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace thesis_project {
namespace wrappers {

PseudoDNNWrapper::PseudoDNNWrapper(int input_size, float support_multiplier, bool rotate_to_upright)
    : input_size_(input_size), support_mult_(support_multiplier), rotate_upright_(rotate_to_upright) {
}

cv::Mat PseudoDNNWrapper::extractPatch(const cv::Mat& image, const cv::KeyPoint& kp) {
    float S = std::max(1.0f, support_mult_ * kp.size);
    // Undo keypoint angle to rotate patch to upright
    float angle = rotate_upright_ ? -kp.angle : 0.0f;
    
    // Simple patch extraction around keypoint
    cv::Point2f center = kp.pt;
    cv::Size2f patch_size(S, S);
    
    // Create rotation matrix
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    
    // Extract rotated patch
    cv::Mat patch;
    cv::warpAffine(image, patch, M, cv::Size(input_size_, input_size_));
    
    return patch;
}

cv::Mat PseudoDNNWrapper::computePseudoCNNFeatures(const cv::Mat& patch) {
    std::vector<cv::Mat> features;
    
    // Simulate conv layer 1: Multi-scale Gaussian filtering
    for (int sigma = 1; sigma <= 3; sigma++) {
        cv::Mat blurred;
        cv::GaussianBlur(patch, blurred, cv::Size(5, 5), sigma);
        
        // Compute gradients (simulates edge detection)
        cv::Mat grad_x, grad_y;
        cv::Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, grad_y, CV_32F, 0, 1, 3);
        
        cv::Mat magnitude;
        cv::magnitude(grad_x, grad_y, magnitude);
        features.push_back(magnitude);
    }
    
    // Simulate conv layer 2: Local binary patterns
    cv::Mat lbp = cv::Mat::zeros(patch.size(), CV_8U);
    for (int i = 1; i < patch.rows - 1; i++) {
        for (int j = 1; j < patch.cols - 1; j++) {
            uint8_t center = patch.at<uint8_t>(i, j);
            uint8_t code = 0;
            
            // 8-connected LBP
            code |= (patch.at<uint8_t>(i-1, j-1) > center) << 7;
            code |= (patch.at<uint8_t>(i-1, j) > center) << 6;
            code |= (patch.at<uint8_t>(i-1, j+1) > center) << 5;
            code |= (patch.at<uint8_t>(i, j+1) > center) << 4;
            code |= (patch.at<uint8_t>(i+1, j+1) > center) << 3;
            code |= (patch.at<uint8_t>(i+1, j) > center) << 2;
            code |= (patch.at<uint8_t>(i+1, j-1) > center) << 1;
            code |= (patch.at<uint8_t>(i, j-1) > center) << 0;
            
            lbp.at<uint8_t>(i, j) = code;
        }
    }
    lbp.convertTo(lbp, CV_32F);
    features.push_back(lbp);
    
    // Spatial pooling: divide into 4x4 grid and compute statistics
    std::vector<float> descriptor;
    for (const auto& feat : features) {
        int grid_size = 4;
        int cell_w = feat.cols / grid_size;
        int cell_h = feat.rows / grid_size;
        
        for (int gi = 0; gi < grid_size; gi++) {
            for (int gj = 0; gj < grid_size; gj++) {
                cv::Rect cell(gj * cell_w, gi * cell_h, cell_w, cell_h);
                cv::Mat cell_data = feat(cell);
                
                // Compute mean and std for each cell
                cv::Scalar mean, stddev;
                cv::meanStdDev(cell_data, mean, stddev);
                descriptor.push_back(static_cast<float>(mean[0]));
                descriptor.push_back(static_cast<float>(stddev[0]));
            }
        }
    }
    
    // Convert to cv::Mat
    cv::Mat result(1, static_cast<int>(descriptor.size()), CV_32F, descriptor.data());
    return result.clone();
}

void PseudoDNNWrapper::initializePCA(const std::vector<cv::Mat>& samples) {
    if (samples.empty()) return;
    
    // Stack samples into matrix
    cv::Mat data(static_cast<int>(samples.size()), samples[0].cols, CV_32F);
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i].copyTo(data.row(static_cast<int>(i)));
    }
    
    // Compute PCA to reduce to 128 dimensions
    pca_(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 128);
    pca_initialized_ = true;
}

cv::Mat PseudoDNNWrapper::extract(const cv::Mat& image, 
                                  const std::vector<cv::KeyPoint>& keypoints,
                                  const DescriptorParams& params) {
    cv::Mat descriptors(static_cast<int>(keypoints.size()), descriptorSize(), CV_32F);
    std::vector<cv::Mat> raw_features;
    
    // Extract features for all keypoints
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::Mat patch = extractPatch(image, keypoints[i]);
        if (patch.channels() > 1) {
            cv::cvtColor(patch, patch, cv::COLOR_BGR2GRAY);
        }
        
        cv::Mat features = computePseudoCNNFeatures(patch);
        raw_features.push_back(features);
    }
    
    // Initialize PCA if not done yet
    if (!pca_initialized_ && !raw_features.empty()) {
        initializePCA(raw_features);
    }
    
    // Apply PCA and normalize
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::Mat desc_row = descriptors.row(static_cast<int>(i));
        
        if (pca_initialized_) {
            cv::Mat projected = pca_.project(raw_features[i]);
            projected.copyTo(desc_row);
        } else {
            // Fallback: just use raw features (truncated/padded to 128)
            int copy_size = std::min(raw_features[i].cols, descriptorSize());
            raw_features[i].colRange(0, copy_size).copyTo(desc_row.colRange(0, copy_size));
        }
        
        // L2 normalize
        cv::normalize(desc_row, desc_row, 1.0, 0.0, cv::NORM_L2);
    }
    
    return descriptors;
}

} // namespace wrappers
} // namespace thesis_project
