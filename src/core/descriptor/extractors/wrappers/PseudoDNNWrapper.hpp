#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
#include <opencv2/opencv.hpp>

namespace thesis_project {
namespace wrappers {

/**
 * @brief Pseudo-DNN descriptor that mimics CNN behavior using OpenCV operations
 * 
 * This descriptor simulates a convolutional neural network using traditional
 * computer vision operations. It serves as a well-documented comparison point
 * for actual DNN descriptors.
 * 
 * Architecture simulation:
 * 1. Multi-scale Gaussian filtering (simulates conv layers)
 * 2. Local binary patterns (simulates learned features) 
 * 3. Spatial pooling (simulates pooling layers)
 * 4. PCA dimensionality reduction (simulates FC layers)
 */
class PseudoDNNWrapper : public IDescriptorExtractor {
private:
    int input_size_ = 32;
    float support_mult_ = 1.0f;
    bool rotate_upright_ = true;
    cv::PCA pca_;
    bool pca_initialized_ = false;
    std::vector<cv::Mat> training_samples_;
    
public:
    explicit PseudoDNNWrapper(int input_size = 32,
                             float support_multiplier = 1.0f, 
                             bool rotate_to_upright = true);

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const thesis_project::DescriptorParams& params = {}) override;

    std::string name() const override { return "LightweightCNN"; }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return DESCRIPTOR_SIFT; }
    
private:
    cv::Mat extractPatch(const cv::Mat& image, const cv::KeyPoint& kp);
    cv::Mat computePseudoCNNFeatures(const cv::Mat& patch);
    void initializePCA(const std::vector<cv::Mat>& samples);
};

} // namespace wrappers
} // namespace thesis_project