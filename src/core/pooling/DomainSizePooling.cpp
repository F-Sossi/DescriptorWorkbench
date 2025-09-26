#include "DomainSizePooling.hpp"
#include "src/core/config/experiment_config.hpp"
#include "keypoints/VanillaSIFT.h"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/pooling/pooling_utils.hpp"
#include "src/core/config/ExperimentConfig.hpp"
#include <algorithm>

namespace thesis_project::pooling {

cv::Mat DomainSizePooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Ptr<cv::Feature2D>& detector,
    const experiment_config& config
) {
    using namespace thesis_project::pooling::utils;

    cv::Mat processedImage = image.channels() > 1 && config.descriptorOptions.descriptorColorSpace == D_BW
        ? [&](){ cv::Mat g; cv::cvtColor(image, g, cv::COLOR_BGR2GRAY); return g; }()
        : image;

    cv::Mat sum;
    double weight_sum = 0.0;
    int expected_rows = -1, expected_cols = -1;

    for (auto scale : config.descriptorOptions.scales) {
        if (scale <= 0.0f) continue; // skip invalid scales

        // Scale only keypoint size; keep image unchanged to preserve geometry
        std::vector<cv::KeyPoint> kps_scaled;
        kps_scaled.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint k = kp;
            k.size = kp.size * scale;
            kps_scaled.push_back(k);
        }

        // Compute descriptors
        cv::Mat desc;
        std::vector<cv::KeyPoint> kps_out = kps_scaled; // allow detector to adjust
        if (auto vanillaSift = std::dynamic_pointer_cast<VanillaSIFT>(detector)) {
            vanillaSift->compute(processedImage, kps_out, desc);
        } else {
            detector->compute(processedImage, kps_out, desc);
        }
        if (desc.empty()) continue;

        // On first successful scale, lock dimensions
        if (expected_rows < 0) {
            expected_rows = desc.rows;
            expected_cols = desc.cols;
            sum = cv::Mat::zeros(expected_rows, expected_cols, desc.type());
        }

        // Shape safety across scales
        if (desc.rows != expected_rows || desc.cols != expected_cols || desc.type() != sum.type()) {
            // Inconsistent shapes across scales: bail out clearly
            return cv::Mat();
        }

        // Optional BEFORE pooling normalization/rooting
        if (config.descriptorOptions.normalizationStage == BEFORE_POOLING) {
            normalizeRows(desc, config.descriptorOptions.normType);
        }
        if (config.descriptorOptions.rootingStage == R_BEFORE_POOLING) {
            // RootSIFT typically expects L1 before sqrt
            normalizeRows(desc, cv::NORM_L1);
            applyRooting(desc);
        }

        // Determine weight (default 1.0). If scale_weights provided and valid, use corresponding weight
        double w = 1.0;
        const auto& ws = config.descriptorOptions.scale_weights;
        const auto& ss = config.descriptorOptions.scales;
        if (!ws.empty() && ws.size() == ss.size()) {
            auto it = std::find(ss.begin(), ss.end(), scale);
            if (it != ss.end()) {
                size_t idx = static_cast<size_t>(std::distance(ss.begin(), it));
                w = std::max(0.0f, ws[idx]);
            }
        } else {
            // Procedural weighting based on legacy mode: 0=uniform,1=triangular,2=gaussian
            if (config.descriptorOptions.scale_weighting_mode == 2) { // gaussian
                double d = std::log(std::max(1e-6, (double)scale));
                double sigma = std::max(1e-6, (double)config.descriptorOptions.scale_weight_sigma);
                w = std::exp(-0.5 * (d / sigma) * (d / sigma));
            } else if (config.descriptorOptions.scale_weighting_mode == 1) { // triangular
                double d = std::abs(std::log(std::max(1e-6, (double)scale)));
                double radius = std::max(1e-6, (double)config.descriptorOptions.scale_weight_sigma) * 2.0;
                w = std::max(0.0, 1.0 - d / radius);
            } else {
                w = 1.0;
            }
        }
        if (w > 0.0) {
            sum += desc * w;
            weight_sum += w;
        }
    }

    if (weight_sum <= 0.0) return cv::Mat();

    // Normalize by sum of weights (average if all 1.0)
    sum.convertTo(sum, sum.type(), 1.0 / weight_sum);

    // AFTER pooling normalization/rooting
    if (config.descriptorOptions.rootingStage == R_AFTER_POOLING) {
        normalizeRows(sum, cv::NORM_L1);
        applyRooting(sum);
    }
    if (config.descriptorOptions.normalizationStage == AFTER_POOLING) {
        normalizeRows(sum, config.descriptorOptions.normType);
    }

    return sum;
}

void DomainSizePooling::applyRooting(cv::Mat& descriptors) const {
    // Apply square root to each descriptor element
    // This is a common technique in computer vision to reduce the influence of large values
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            float& val = descriptors.at<float>(i, j);
            if (val >= 0) {
                val = std::sqrt(val);
            } else {
                val = -std::sqrt(-val); // Handle negative values
            }
        }
    }
}

// New interface overload: pool using IDescriptorExtractor
cv::Mat DomainSizePooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
thesis_project::IDescriptorExtractor& extractor,
const experiment_config& config
) {
    using namespace thesis_project::pooling::utils;

    cv::Mat processedImage = image.channels() > 1 && config.descriptorOptions.descriptorColorSpace == D_BW
        ? [&](){ cv::Mat g; cv::cvtColor(image, g, cv::COLOR_BGR2GRAY); return g; }()
        : image;

    cv::Mat sum;
    double weight_sum = 0.0;
    int expected_rows = -1, expected_cols = -1;

    for (auto scale : config.descriptorOptions.scales) {
        if (scale <= 0.0f) continue;
        std::vector<cv::KeyPoint> kps_scaled;
        kps_scaled.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint k = kp;
            k.size = kp.size * scale;
            kps_scaled.push_back(k);
        }

        // Legacy config doesn't have device settings - use default params
        thesis_project::DescriptorParams params; // defaults to device="auto"
        cv::Mat desc = extractor.extract(processedImage, kps_scaled, params);
        if (desc.empty()) continue;

        if (expected_rows < 0) {
            expected_rows = desc.rows;
            expected_cols = desc.cols;
            sum = cv::Mat::zeros(expected_rows, expected_cols, desc.type());
        }
        if (desc.rows != expected_rows || desc.cols != expected_cols || desc.type() != sum.type()) {
            return cv::Mat();
        }

        if (config.descriptorOptions.normalizationStage == BEFORE_POOLING) {
            normalizeRows(desc, config.descriptorOptions.normType);
        }
        if (config.descriptorOptions.rootingStage == R_BEFORE_POOLING) {
            normalizeRows(desc, cv::NORM_L1);
            applyRooting(desc);
        }

        double w = 1.0;
        const auto& ws = config.descriptorOptions.scale_weights;
        const auto& ss = config.descriptorOptions.scales;
        if (!ws.empty() && ws.size() == ss.size()) {
            auto it = std::find(ss.begin(), ss.end(), scale);
            if (it != ss.end()) {
                size_t idx = static_cast<size_t>(std::distance(ss.begin(), it));
                w = std::max(0.0f, ws[idx]);
            }
        } else {
            if (config.descriptorOptions.scale_weighting_mode == 2) { // gaussian
                double d = std::log(std::max(1e-6, (double)scale));
                double sigma = std::max(1e-6, (double)config.descriptorOptions.scale_weight_sigma);
                w = std::exp(-0.5 * (d / sigma) * (d / sigma));
            } else if (config.descriptorOptions.scale_weighting_mode == 1) { // triangular
                double d = std::abs(std::log(std::max(1e-6, (double)scale)));
                double radius = std::max(1e-6, (double)config.descriptorOptions.scale_weight_sigma) * 2.0;
                w = std::max(0.0, 1.0 - d / radius);
            } else {
                w = 1.0;
            }
        }
        if (w > 0.0) {
            sum += desc * w;
            weight_sum += w;
        }
    }

    if (weight_sum <= 0.0) return cv::Mat();

    sum.convertTo(sum, sum.type(), 1.0 / weight_sum);

    if (config.descriptorOptions.rootingStage == R_AFTER_POOLING) {
        normalizeRows(sum, cv::NORM_L1);
        applyRooting(sum);
    }
    if (config.descriptorOptions.normalizationStage == AFTER_POOLING) {
        normalizeRows(sum, config.descriptorOptions.normType);
    }

    return sum;
}

// New-config overload: use descriptor params from YAML new config
cv::Mat DomainSizePooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    using namespace thesis_project::pooling::utils;
    const auto& params = descCfg.params;

    if (params.scales.empty()) {
        // No scales means act like NoPooling
        cv::Mat d = extractor.extract(image, keypoints, params);
        // Optional normalize after pooling flag applies
        if (params.normalize_after_pooling) normalizeRows(d, params.norm_type);
        return d;
    }

    // Prepare accumulators
    cv::Mat acc;
    double weight_sum = 0.0;
    const bool use_weights = !params.scale_weights.empty();

    for (size_t i = 0; i < params.scales.size(); ++i) {
        float alpha = params.scales[i];
        // Scale image by alpha around original resolution
        cv::Mat processedImage;
        if (std::abs(alpha - 1.0f) < 1e-6) {
            processedImage = image;
        } else {
            cv::resize(image, processedImage, cv::Size(), alpha, alpha, cv::INTER_LINEAR);
        }

        // Scale keypoints by alpha
        std::vector<cv::KeyPoint> kps_scaled = keypoints;
        for (auto& kp : kps_scaled) {
            kp.pt.x *= alpha; kp.pt.y *= alpha; kp.size *= alpha;
        }

        // Extract per-scale descriptors
        cv::Mat desc = extractor.extract(processedImage, kps_scaled, params);

        // Normalize before pooling if requested
        if (params.normalize_before_pooling) normalizeRows(desc, params.norm_type);

        // Weight
        double w = 1.0;
        if (use_weights) {
            w = static_cast<double>(params.scale_weights[i]);
        } else {
            // Procedural weighting
            switch (params.scale_weighting) {
                case thesis_project::ScaleWeighting::GAUSSIAN: {
                    double sigma = params.scale_weight_sigma;
                    double x = std::log(alpha);
                    w = std::exp(-0.5 * (x*x) / (sigma*sigma));
                    break;
                }
                case thesis_project::ScaleWeighting::TRIANGULAR: {
                    double r = params.scale_weight_sigma; // treat as radius proxy
                    double d = std::abs(std::log(alpha));
                    w = std::max(0.0, 1.0 - d / r);
                    break;
                }
                case thesis_project::ScaleWeighting::UNIFORM:
                default: w = 1.0; break;
            }
        }

        // Accumulate
        if (acc.empty()) {
            acc = cv::Mat::zeros(desc.size(), desc.type());
        }
        acc += desc * w;
        weight_sum += w;
    }

    // Average
    if (weight_sum > 0.0) acc /= weight_sum;

    // Normalize after pooling if requested
    if (params.normalize_after_pooling) normalizeRows(acc, params.norm_type);

    return acc;
}

} // namespace thesis_project::pooling
