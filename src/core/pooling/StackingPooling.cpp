#include "StackingPooling.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/VanillaSIFT.h"
#include <iostream>
#include "src/core/pooling/pooling_utils.hpp"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include <opencv2/imgproc.hpp>
#include "src/core/config/ExperimentConfig.hpp"

namespace thesis_project::pooling {

cv::Mat StackingPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Ptr<cv::Feature2D>& detector,
    const experiment_config& config
) {
    // Validate that we have a secondary detector for stacking
    if (!config.detector2) {
        std::cerr << "[ERROR] No secondary detector configured for stacking" << std::endl;
        return cv::Mat();
    }

    // Prepare images for each descriptor's color space requirements
    cv::Mat image1 = prepareImageForColorSpace(image, config.descriptorOptions.descriptorColorSpace);
    cv::Mat image2 = prepareImageForColorSpace(image, config.descriptorOptions.descriptorColorSpace2);

    // Compute first descriptor (capture potentially adjusted keypoints)
    std::vector<cv::KeyPoint> keypoints1 = keypoints;
    cv::Mat descriptor1 = computeDescriptorsWithDetector(image1, keypoints1, detector);

    // Compute second descriptor  (capture potentially adjusted keypoints)
    std::vector<cv::KeyPoint> keypoints2 = keypoints;
    cv::Mat descriptor2 = computeDescriptorsWithDetector(image2, keypoints2, config.detector2);

    // Validate descriptors before concatenation
    if (descriptor1.empty()) {
        std::cerr << "[ERROR] First descriptor is empty in stacking" << std::endl;
        return cv::Mat();
    }
    if (descriptor2.empty()) {
        std::cerr << "[ERROR] Second descriptor is empty in stacking" << std::endl;
        return cv::Mat();
    }
    
    // Check keypoint alignment and dimensional compatibility
    if (descriptor1.rows != descriptor2.rows) {
        std::cerr << "[ERROR] Descriptor row mismatch: " << descriptor1.rows
                  << " vs " << descriptor2.rows << std::endl;
        return cv::Mat();
    }
    // Enforce that keypoints correspond (within tolerance)
    const double eps = 0.5; // pixels
    if (keypoints1.size() != keypoints2.size()) {
        std::cerr << "[ERROR] Keypoint count mismatch in stacking: " << keypoints1.size()
                  << " vs " << keypoints2.size() << std::endl;
        return cv::Mat();
    }
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        if (std::abs(keypoints1[i].pt.x - keypoints2[i].pt.x) > eps ||
            std::abs(keypoints1[i].pt.y - keypoints2[i].pt.y) > eps) {
            std::cerr << "[ERROR] Keypoint misalignment at row " << i << " in stacking" << std::endl;
            return cv::Mat();
        }
    }

    using namespace thesis_project::pooling::utils;
    // BEFORE pooling normalization/rooting per component
    if (config.descriptorOptions.normalizationStage == BEFORE_POOLING) {
        normalizeRows(descriptor1, config.descriptorOptions.normType);
        normalizeRows(descriptor2, config.descriptorOptions.normType);
    }
    if (config.descriptorOptions.rootingStage == R_BEFORE_POOLING) {
        normalizeRows(descriptor1, cv::NORM_L1);
        normalizeRows(descriptor2, cv::NORM_L1);
        applyRooting(descriptor1);
        applyRooting(descriptor2);
    }

    // Horizontally concatenate descriptors
    cv::Mat stackedDescriptor;
    cv::hconcat(descriptor1, descriptor2, stackedDescriptor);

    // AFTER pooling normalization/rooting on concatenated vector
    if (config.descriptorOptions.rootingStage == R_AFTER_POOLING) {
        normalizeRows(stackedDescriptor, cv::NORM_L1);
        applyRooting(stackedDescriptor);
    }
    if (config.descriptorOptions.normalizationStage == AFTER_POOLING) {
        normalizeRows(stackedDescriptor, config.descriptorOptions.normType);
    }

    return stackedDescriptor;
}

cv::Mat StackingPooling::prepareImageForColorSpace(const cv::Mat& sourceImage, int targetColorSpace) const {
    cv::Mat targetImage = sourceImage.clone();
    
    if (targetColorSpace == D_BW && sourceImage.channels() > 1) {
        // Convert to grayscale for BW descriptors
        cv::cvtColor(sourceImage, targetImage, cv::COLOR_BGR2GRAY);
    } else if (targetColorSpace == D_COLOR && sourceImage.channels() == 1) {
        // Convert grayscale to color for color descriptors  
        cv::cvtColor(sourceImage, targetImage, cv::COLOR_GRAY2BGR);
    }
    
    return targetImage;
}

cv::Mat StackingPooling::computeDescriptorsWithDetector(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Ptr<cv::Feature2D>& detector
) const {
    cv::Mat descriptors;
    // detector->compute may adjust keypoints in-place
    // Handle VanillaSIFT vs standard Feature2D interface
    if (auto vanillaSift = std::dynamic_pointer_cast<VanillaSIFT>(detector)) {
        vanillaSift->compute(image, keypoints, descriptors);
    } else {
        detector->compute(image, keypoints, descriptors);
    }

    return descriptors;
}

cv::Mat StackingPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const experiment_config& config
) {
    // Validate that we have a secondary descriptor configured
    if (config.descriptorOptions.descriptorType2 == NO_DESCRIPTOR) {
        std::cerr << "[ERROR] stacking requires secondary descriptor" << std::endl;
        return cv::Mat();
    }

    // Prepare images according to desired color spaces
    cv::Mat image1 = prepareImageForColorSpace(image, config.descriptorOptions.descriptorColorSpace);
    cv::Mat image2 = prepareImageForColorSpace(image, config.descriptorOptions.descriptorColorSpace2);

    // Copy keypoints for each branch (extractors may adjust keypoints)
    std::vector<cv::KeyPoint> keypoints1 = keypoints;
    std::vector<cv::KeyPoint> keypoints2 = keypoints;

    // Compute first descriptor using provided extractor
    // Legacy config doesn't have device settings - use default params
    thesis_project::DescriptorParams params; // defaults to device="auto"
    cv::Mat descriptor1 = extractor.extract(image1, keypoints1, params);

    // Create secondary extractor by cloning config with descriptorType2
    experiment_config cfg2 = config;
    cfg2.descriptorOptions.descriptorType = config.descriptorOptions.descriptorType2;
    // Update image type/color space for second branch based on config
    // (wrappers use input image; we prep image2 accordingly)
    auto extractor2 = thesis_project::factories::DescriptorFactory::create(cfg2);
    cv::Mat descriptor2 = extractor2->extract(image2, keypoints2, params);

    // Validate descriptors before concatenation
    if (descriptor1.empty() || descriptor2.empty()) {
        std::cerr << "[ERROR] Empty component in stacking" << std::endl;
        return cv::Mat();
    }
    if (descriptor1.rows != descriptor2.rows) {
        std::cerr << "[ERROR] Descriptor row mismatch: " << descriptor1.rows
                  << " vs " << descriptor2.rows << std::endl;
        return cv::Mat();
    }

    // Enforce approximate keypoint alignment
    const double eps = 0.5;
    if (keypoints1.size() != keypoints2.size()) {
        std::cerr << "[ERROR] Keypoint count mismatch in stacking: " << keypoints1.size()
                  << " vs " << keypoints2.size() << std::endl;
        return cv::Mat();
    }
    for (size_t i = 0; i < keypoints1.size(); ++i) {
        if (std::abs(keypoints1[i].pt.x - keypoints2[i].pt.x) > eps ||
            std::abs(keypoints1[i].pt.y - keypoints2[i].pt.y) > eps) {
            std::cerr << "[ERROR] Keypoint misalignment at row " << i << " in stacking" << std::endl;
            return cv::Mat();
        }
    }

    using namespace thesis_project::pooling::utils;
    // BEFORE pooling normalization/rooting per component
    if (config.descriptorOptions.normalizationStage == BEFORE_POOLING) {
        normalizeRows(descriptor1, config.descriptorOptions.normType);
        normalizeRows(descriptor2, config.descriptorOptions.normType);
    }
    if (config.descriptorOptions.rootingStage == R_BEFORE_POOLING) {
        normalizeRows(descriptor1, cv::NORM_L1);
        normalizeRows(descriptor2, cv::NORM_L1);
        applyRooting(descriptor1);
        applyRooting(descriptor2);
    }

    // Concatenate
    cv::Mat stackedDescriptor;
    cv::hconcat(descriptor1, descriptor2, stackedDescriptor);

    // AFTER pooling normalization/rooting
    if (config.descriptorOptions.rootingStage == R_AFTER_POOLING) {
        normalizeRows(stackedDescriptor, cv::NORM_L1);
        applyRooting(stackedDescriptor);
    }
    if (config.descriptorOptions.normalizationStage == AFTER_POOLING) {
        normalizeRows(stackedDescriptor, config.descriptorOptions.normType);
    }

    return stackedDescriptor;
}

cv::Mat StackingPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    using namespace thesis_project::pooling::utils;
    // Determine color space from params
    int color1 = descCfg.params.use_color ? D_COLOR : D_BW;
    // Secondary color choice heuristic
    int color2;
    switch (descCfg.params.secondary_descriptor) {
        case thesis_project::DescriptorType::RGBSIFT:
        case thesis_project::DescriptorType::HoNC:
            color2 = D_COLOR; break;
        default:
            color2 = D_BW; break;
    }

    cv::Mat image1 = prepareImageForColorSpace(image, color1);
    cv::Mat image2 = prepareImageForColorSpace(image, color2);

    std::vector<cv::KeyPoint> kps1 = keypoints;
    std::vector<cv::KeyPoint> kps2 = keypoints;

    cv::Mat d1 = extractor.extract(image1, kps1, descCfg.params);

    // Create secondary extractor from descriptor type
    auto ext2 = thesis_project::factories::DescriptorFactory::create(descCfg.params.secondary_descriptor);
    cv::Mat d2 = ext2->extract(image2, kps2, descCfg.params);

    if (d1.empty() || d2.empty()) return cv::Mat();
    if (d1.rows != d2.rows) return cv::Mat();
    if (kps1.size() != kps2.size()) return cv::Mat();
    const double eps = 0.5;
    for (size_t i = 0; i < kps1.size(); ++i) {
        if (std::abs(kps1[i].pt.x - kps2[i].pt.x) > eps || std::abs(kps1[i].pt.y - kps2[i].pt.y) > eps) {
            return cv::Mat();
        }
    }

    // BEFORE pooling normalization/rooting per component
    if (descCfg.params.normalize_before_pooling) {
        normalizeRows(d1, descCfg.params.norm_type);
        normalizeRows(d2, descCfg.params.norm_type);
    }

    cv::Mat stacked;
    cv::hconcat(d1, d2, stacked);

    // AFTER pooling normalization
    if (descCfg.params.normalize_after_pooling) {
        normalizeRows(stacked, descCfg.params.norm_type);
    }

    return stacked;
}

} // namespace thesis_project::pooling
