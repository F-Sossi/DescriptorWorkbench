#include "StackingPooling.hpp"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "src/core/pooling/pooling_utils.hpp"
#include <algorithm>

namespace thesis_project::pooling {

namespace {
cv::Mat convertForDescriptor(const cv::Mat& image, bool useColor) {
    if (useColor) {
        if (image.channels() == 1) {
            cv::Mat color;
            cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
            return color;
        }
        return image;
    }

    if (image.channels() > 1) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return image;
}

bool descriptorRequiresColor(thesis_project::DescriptorType type) {
    switch (type) {
        case thesis_project::DescriptorType::RGBSIFT:
        case thesis_project::DescriptorType::HoNC:
            return true;
        default:
            return false;
    }
}
}

cv::Mat StackingPooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    using namespace thesis_project::pooling::utils;

    const auto& params = descCfg.params;
    auto secondaryType = params.secondary_descriptor;
    if (secondaryType == thesis_project::DescriptorType::NONE) {
        // Fallback to primary descriptor only
        cv::Mat descriptors = extractor.extract(convertForDescriptor(image, params.use_color), keypoints, params);
        if (params.normalize_after_pooling) {
            normalizeRows(descriptors, params.norm_type);
        }
        if (params.rooting_stage == thesis_project::RootingStage::R_AFTER_POOLING) {
            normalizeRows(descriptors, cv::NORM_L1);
            applyRooting(descriptors);
        }
        return descriptors;
    }

    bool primaryColor = params.use_color;
    bool secondaryColor = descriptorRequiresColor(secondaryType);

    cv::Mat imagePrimary = prepareImageForColorSpace(image, primaryColor);

    std::vector<cv::KeyPoint> kpsPrimary = keypoints;
    std::vector<cv::KeyPoint> kpsSecondary = keypoints;

    cv::Mat descriptorsPrimary = extractor.extract(imagePrimary, kpsPrimary, params);
    if (descriptorsPrimary.empty()) {
        return cv::Mat();
    }

    cv::Mat descriptorsSecondary = computeSecondaryDescriptors(image, kpsSecondary, secondaryType, secondaryColor, params);
    if (descriptorsSecondary.empty()) {
        return cv::Mat();
    }

    if (descriptorsPrimary.rows != descriptorsSecondary.rows) {
        return cv::Mat();
    }

    const bool rootBefore = params.rooting_stage == thesis_project::RootingStage::R_BEFORE_POOLING;
    if (params.normalize_before_pooling) {
        normalizeRows(descriptorsPrimary, params.norm_type);
        normalizeRows(descriptorsSecondary, params.norm_type);
    }
    if (rootBefore) {
        normalizeRows(descriptorsPrimary, cv::NORM_L1);
        normalizeRows(descriptorsSecondary, cv::NORM_L1);
        applyRooting(descriptorsPrimary);
        applyRooting(descriptorsSecondary);
    }

    const float weight = params.stacking_weight;
    if (weight > 0.0f && weight < 1.0f) {
        descriptorsSecondary *= weight;
    }

    cv::Mat stacked;
    cv::hconcat(descriptorsPrimary, descriptorsSecondary, stacked);

    if (params.rooting_stage == thesis_project::RootingStage::R_AFTER_POOLING) {
        normalizeRows(stacked, cv::NORM_L1);
        applyRooting(stacked);
    }

    if (params.normalize_after_pooling) {
        normalizeRows(stacked, params.norm_type);
    }

    return stacked;
}

cv::Mat StackingPooling::prepareImageForColorSpace(const cv::Mat& sourceImage, bool useColor) const {
    return convertForDescriptor(sourceImage, useColor);
}

cv::Mat StackingPooling::computeSecondaryDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::DescriptorType descriptorType,
    bool useColor,
    const thesis_project::DescriptorParams& params
) const {
    auto extractor = thesis_project::factories::DescriptorFactory::create(descriptorType);
    if (!extractor) {
        return cv::Mat();
    }
    cv::Mat prepped = convertForDescriptor(image, useColor);
    return extractor->extract(prepped, keypoints, params);
}

} // namespace thesis_project::pooling
