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

        // Pyramid-aware pooling: apply appropriate blur for each scale
        // Scale-space theory: smaller scales need less blur, larger scales need more blur
        // Base sigma for SIFT is typically 1.6
        const double base_sigma = 1.6;
        double sigma = base_sigma * scale;

        // Apply Gaussian blur appropriate for this scale
        cv::Mat blurred_image;
        if (std::abs(scale - 1.0) < 1e-6) {
            // At scale 1.0, use original blur (or apply base sigma)
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), base_sigma);
        } else if (scale < 1.0) {
            // Smaller scale -> less blur (sharper features)
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), sigma);
        } else {
            // Larger scale -> more blur (coarser features)
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), sigma);
        }

        // Scale keypoint sizes to sample appropriate region size
        std::vector<cv::KeyPoint> kps_scaled;
        kps_scaled.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint k = kp;
            k.size = kp.size * scale;
            kps_scaled.push_back(k);
        }

        // Compute descriptors from the appropriately blurred image
        cv::Mat desc;
        std::vector<cv::KeyPoint> kps_out = kps_scaled; // allow detector to adjust
        if (auto vanillaSift = std::dynamic_pointer_cast<VanillaSIFT>(detector)) {
            vanillaSift->compute(blurred_image, kps_out, desc);
        } else {
            detector->compute(blurred_image, kps_out, desc);
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

        // Pyramid-aware pooling: apply appropriate blur for each scale
        const double base_sigma = 1.6;
        double sigma = base_sigma * scale;
        cv::Mat blurred_image;
        if (std::abs(scale - 1.0) < 1e-6) {
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), base_sigma);
        } else if (scale < 1.0) {
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), sigma);
        } else {
            cv::GaussianBlur(processedImage, blurred_image, cv::Size(0, 0), sigma);
        }

        std::vector<cv::KeyPoint> kps_scaled;
        kps_scaled.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint k = kp;
            k.size = kp.size * scale;
            kps_scaled.push_back(k);
        }

        // Legacy config doesn't have device settings - use default params
        thesis_project::DescriptorParams params; // defaults to device="auto"
        cv::Mat desc = extractor.extract(blurred_image, kps_scaled, params);
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

    // TRUE PYRAMID-AWARE POOLING: Build pyramid once, sample from appropriate levels
    // This matches DSPSIFT's approach of reusing the same scale-space structure

    // Build SIFT-style Gaussian pyramid once
    const int nOctaves = 4;
    const int nScalesPerOctave = 3;
    const double sigma0 = 1.6;
    std::vector<std::vector<cv::Mat>> pyramid;
    buildGaussianPyramid(image, pyramid, nOctaves, nScalesPerOctave, sigma0);

    // Collect descriptors from all scales
    std::vector<cv::Mat> descriptors_per_scale;
    std::vector<double> weights;
    const bool use_weights = !params.scale_weights.empty();

    for (size_t i = 0; i < params.scales.size(); ++i) {
        float alpha = params.scales[i];

        // Find the appropriate pyramid level for this scale factor
        auto [octave, scale] = findPyramidLevel(alpha, nScalesPerOctave);

        // Safety check: ensure pyramid level exists
        if (octave >= static_cast<int>(pyramid.size()) ||
            scale >= static_cast<int>(pyramid[octave].size())) {
            continue; // Skip invalid pyramid levels
        }

        // Get the pre-built pyramid level
        const cv::Mat& pyramid_level = pyramid[octave][scale];

        // Transform keypoints to pyramid space
        // Octave 0 = original resolution, octave 1 = 1/2 resolution, etc.
        double scale_factor = 1.0 / (1 << octave); // 2^(-octave)
        std::vector<cv::KeyPoint> kps_scaled = keypoints;
        for (auto& kp : kps_scaled) {
            kp.pt.x *= scale_factor;
            kp.pt.y *= scale_factor;
            kp.size *= alpha * scale_factor; // Scale size for domain pooling AND pyramid level
        }

        // Extract descriptors from the pre-built pyramid level
        cv::Mat desc = extractor.extract(pyramid_level, kps_scaled, params);

        // Normalize before pooling if requested
        if (params.normalize_before_pooling) normalizeRows(desc, params.norm_type);

        // Calculate weight for this scale
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

        descriptors_per_scale.push_back(desc);
        weights.push_back(w);
    }

    // Aggregate descriptors using configured method
    cv::Mat acc;
    aggregateDescriptors(descriptors_per_scale, weights, acc, params.pooling_aggregation);

    // Normalize after pooling if requested
    if (params.normalize_after_pooling) normalizeRows(acc, params.norm_type);

    return acc;
}

void DomainSizePooling::buildGaussianPyramid(const cv::Mat& image,
                                              std::vector<std::vector<cv::Mat>>& pyramid,
                                              int nOctaves,
                                              int nScalesPerOctave,
                                              double sigma0) const {
    // Build SIFT-style Gaussian pyramid matching VanillaSIFT::buildGaussianPyramid
    // This creates a multi-octave, multi-scale structure like SIFT uses internally

    pyramid.clear();
    pyramid.resize(nOctaves);

    const int nScales = nScalesPerOctave + 3; // Extra scales for DoG computation
    const double k = std::pow(2.0, 1.0 / nScalesPerOctave); // Scale multiplier per level

    for (int o = 0; o < nOctaves; o++) {
        pyramid[o].resize(nScales);

        // Base image for this octave
        cv::Mat octaveBase;
        if (o == 0) {
            octaveBase = image.clone();
        } else {
            // Downsample from previous octave's middle scale
            cv::pyrDown(pyramid[o-1][nScalesPerOctave], octaveBase);
        }

        // First scale is the base
        pyramid[o][0] = octaveBase.clone();

        // Build remaining scales in this octave
        for (int s = 1; s < nScales; s++) {
            double sigma_prev = sigma0 * std::pow(k, s - 1);
            double sigma_curr = sigma0 * std::pow(k, s);

            // Incremental blur: only apply the DIFFERENCE in sigma
            double sigma_diff = std::sqrt(sigma_curr * sigma_curr - sigma_prev * sigma_prev);

            cv::GaussianBlur(pyramid[o][s-1], pyramid[o][s], cv::Size(0, 0), sigma_diff);
        }
    }
}

std::pair<int, int> DomainSizePooling::findPyramidLevel(float scaleFactor, int nScalesPerOctave) const {
    // Map scale factor to pyramid coordinates (octave, scale)
    // scaleFactor < 1.0 means finer details → higher octaves or later scales
    // scaleFactor > 1.0 means coarser features → lower octaves or earlier scales

    if (std::abs(scaleFactor - 1.0f) < 1e-6) {
        // Scale 1.0 = base octave, base scale
        return {0, 0};
    }

    // Convert scale factor to octave space
    // log2(scaleFactor) gives the octave offset
    double octave_float = std::log2(scaleFactor);
    int octave = static_cast<int>(std::floor(octave_float));

    // Remaining fractional part maps to intra-octave scale
    double scale_fraction = octave_float - octave;
    int scale = static_cast<int>(std::round(scale_fraction * nScalesPerOctave));

    // Clamp to valid ranges
    octave = std::max(0, std::min(3, octave)); // Assuming 4 octaves
    scale = std::max(0, std::min(nScalesPerOctave + 2, scale));

    return {octave, scale};
}

void DomainSizePooling::aggregateDescriptors(
    const std::vector<cv::Mat>& descriptors_per_scale,
    const std::vector<double>& weights,
    cv::Mat& output,
    thesis_project::PoolingAggregation aggregation) const {

    if (descriptors_per_scale.empty()) {
        output = cv::Mat();
        return;
    }

    const int rows = descriptors_per_scale[0].rows;
    const int cols = descriptors_per_scale[0].cols;
    const int type = descriptors_per_scale[0].type();

    switch (aggregation) {
        case thesis_project::PoolingAggregation::AVERAGE:
        case thesis_project::PoolingAggregation::WEIGHTED_AVG: {
            output = cv::Mat::zeros(rows, cols, type);
            double weight_sum = 0.0;
            for (size_t i = 0; i < descriptors_per_scale.size(); ++i) {
                double w = (i < weights.size()) ? weights[i] : 1.0;
                output += descriptors_per_scale[i] * w;
                weight_sum += w;
            }
            if (weight_sum > 0.0) output /= weight_sum;
            break;
        }

        case thesis_project::PoolingAggregation::MAX: {
            output = descriptors_per_scale[0].clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); ++i) {
                cv::max(output, descriptors_per_scale[i], output);
            }
            break;
        }

        case thesis_project::PoolingAggregation::MIN: {
            output = descriptors_per_scale[0].clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); ++i) {
                cv::min(output, descriptors_per_scale[i], output);
            }
            break;
        }

        case thesis_project::PoolingAggregation::CONCATENATE: {
            // Horizontal concatenation (increases dimensionality)
            cv::hconcat(descriptors_per_scale, output);
            break;
        }

        default:
            output = descriptors_per_scale[0].clone();
            break;
    }
}

} // namespace thesis_project::pooling
