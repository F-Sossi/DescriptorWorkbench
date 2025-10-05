#include "DomainSizePooling.hpp"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "src/core/pooling/pooling_utils.hpp"
#include "src/core/config/ExperimentConfig.hpp"
#include <algorithm>
#include <cmath>

namespace thesis_project::pooling {

namespace {
cv::Mat maybeConvertToGrayscale(const cv::Mat& image, bool useColor) {
    if (useColor || image.channels() == 1) {
        return image;
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}
}

cv::Mat DomainSizePooling::computeDescriptors(
    const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints,
    thesis_project::IDescriptorExtractor& extractor,
    const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg
) {
    using namespace thesis_project::pooling::utils;
    const auto& params = descCfg.params;

    // Short-circuit: no pooling requested -> simple extraction
    if (params.scales.empty()) {
        cv::Mat descriptors = extractor.extract(image, keypoints, params);
        if (params.rooting_stage == thesis_project::RootingStage::R_AFTER_POOLING) {
            if (!descriptors.empty()) {
                normalizeRows(descriptors, cv::NORM_L1);
                applyRooting(descriptors);
            }
        }
        if (params.normalize_after_pooling) {
            normalizeRows(descriptors, params.norm_type);
        }
        return descriptors;
    }

    const bool root_before = params.rooting_stage == thesis_project::RootingStage::R_BEFORE_POOLING;
    const bool root_after  = params.rooting_stage == thesis_project::RootingStage::R_AFTER_POOLING;

    cv::Mat workingImage = maybeConvertToGrayscale(image, params.use_color);

    // Build SIFT-style Gaussian pyramid once
    const int nOctaves = 4;
    const int nScalesPerOctave = 3;
    const double sigma0 = 1.6;
    std::vector<std::vector<cv::Mat>> pyramid;
    buildGaussianPyramid(workingImage, pyramid, nOctaves, nScalesPerOctave, sigma0);

    std::vector<cv::Mat> descriptorsPerScale;
    std::vector<double> weights;
    descriptorsPerScale.reserve(params.scales.size());
    weights.reserve(params.scales.size());

    const bool hasExplicitWeights = !params.scale_weights.empty() &&
                                    params.scale_weights.size() == params.scales.size();

    for (size_t i = 0; i < params.scales.size(); ++i) {
        float alpha = params.scales[i];
        if (alpha <= 0.0f) continue;

        auto [octave, scaleIndex] = findPyramidLevel(alpha, nScalesPerOctave);
        if (octave < 0 || octave >= static_cast<int>(pyramid.size())) continue;
        if (scaleIndex < 0 || scaleIndex >= static_cast<int>(pyramid[octave].size())) continue;

        const cv::Mat& levelImage = pyramid[octave][scaleIndex];
        double scaleFactor = 1.0 / (1 << octave);

        std::vector<cv::KeyPoint> scaledKeypoints;
        scaledKeypoints.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            cv::KeyPoint adjusted = kp;
            adjusted.pt.x *= static_cast<float>(scaleFactor);
            adjusted.pt.y *= static_cast<float>(scaleFactor);
            adjusted.size *= alpha * static_cast<float>(scaleFactor);
            scaledKeypoints.push_back(adjusted);
        }

        cv::Mat descriptors = extractor.extract(levelImage, scaledKeypoints, params);
        if (descriptors.empty()) continue;

        if (params.normalize_before_pooling) {
            normalizeRows(descriptors, params.norm_type);
        }
        if (root_before) {
            normalizeRows(descriptors, cv::NORM_L1);
            applyRooting(descriptors);
        }

        double w = 1.0;
        if (hasExplicitWeights) {
            w = static_cast<double>(params.scale_weights[i]);
        } else {
            switch (params.scale_weighting) {
                case thesis_project::ScaleWeighting::GAUSSIAN: {
                    double sigma = std::max(1e-6f, params.scale_weight_sigma);
                    double x = std::log(static_cast<double>(alpha));
                    w = std::exp(-0.5 * (x * x) / (sigma * sigma));
                    break;
                }
                case thesis_project::ScaleWeighting::TRIANGULAR: {
                    double radius = std::max(1e-6f, params.scale_weight_sigma);
                    double d = std::abs(std::log(static_cast<double>(alpha)));
                    w = std::max(0.0, 1.0 - d / radius);
                    break;
                }
                case thesis_project::ScaleWeighting::UNIFORM:
                default:
                    w = 1.0;
                    break;
            }
        }

        descriptorsPerScale.push_back(std::move(descriptors));
        weights.push_back(w);
    }

    if (descriptorsPerScale.empty()) {
        return cv::Mat();
    }

    cv::Mat aggregated;
    aggregateDescriptors(descriptorsPerScale, weights, aggregated, params.pooling_aggregation);
    if (aggregated.empty()) {
        return aggregated;
    }

    if (root_after) {
        normalizeRows(aggregated, cv::NORM_L1);
        applyRooting(aggregated);
    }

    if (params.normalize_after_pooling) {
        normalizeRows(aggregated, params.norm_type);
    }

    return aggregated;
}

void DomainSizePooling::applyRooting(cv::Mat& descriptors) const {
    for (int i = 0; i < descriptors.rows; ++i) {
        float* ptr = descriptors.ptr<float>(i);
        for (int j = 0; j < descriptors.cols; ++j) {
            float val = ptr[j];
            ptr[j] = val >= 0.f ? std::sqrt(val) : -std::sqrt(-val);
        }
    }
}

void DomainSizePooling::buildGaussianPyramid(const cv::Mat& image,
                                              std::vector<std::vector<cv::Mat>>& pyramid,
                                              int nOctaves,
                                              int nScalesPerOctave,
                                              double sigma0) const {
    pyramid.clear();
    pyramid.resize(nOctaves);

    const int nScales = nScalesPerOctave + 3;
    const double k = std::pow(2.0, 1.0 / nScalesPerOctave);

    for (int o = 0; o < nOctaves; ++o) {
        pyramid[o].resize(nScales);

        cv::Mat octaveBase;
        if (o == 0) {
            octaveBase = image.clone();
        } else {
            cv::pyrDown(pyramid[o - 1][nScalesPerOctave], octaveBase);
        }

        pyramid[o][0] = octaveBase.clone();

        for (int s = 1; s < nScales; ++s) {
            double sigmaPrev = sigma0 * std::pow(k, s - 1);
            double sigmaCurr = sigma0 * std::pow(k, s);
            double sigmaDiff = std::sqrt(sigmaCurr * sigmaCurr - sigmaPrev * sigmaPrev);

            cv::GaussianBlur(pyramid[o][s - 1], pyramid[o][s], cv::Size(0, 0), sigmaDiff);
        }
    }
}

std::pair<int, int> DomainSizePooling::findPyramidLevel(float scaleFactor, int nScalesPerOctave) const {
    if (scaleFactor <= 0.0f) {
        return {0, 0};
    }

    double logScale = std::log(scaleFactor) / std::log(2.0);
    int octave = static_cast<int>(std::floor(logScale));
    double fractional = logScale - octave;
    int scale = static_cast<int>(std::round(fractional * nScalesPerOctave));

    octave = std::max(0, octave);
    scale = std::max(0, scale);

    return {octave, scale};
}

void DomainSizePooling::aggregateDescriptors(const std::vector<cv::Mat>& descriptors_per_scale,
                                             const std::vector<double>& weights,
                                             cv::Mat& output,
                                             thesis_project::PoolingAggregation aggregation) const {
    using namespace thesis_project::pooling::utils;

    if (descriptors_per_scale.empty()) {
        output.release();
        return;
    }

    const cv::Mat& reference = descriptors_per_scale.front();
    const int rows = reference.rows;
    const int cols = reference.cols;
    const int type = reference.type();

    auto ensureSameShape = [&](const cv::Mat& mat) {
        return mat.rows == rows && mat.cols == cols && mat.type() == type;
    };

    switch (aggregation) {
        case thesis_project::PoolingAggregation::CONCATENATE: {
            cv::hconcat(descriptors_per_scale, output);
            return;
        }
        case thesis_project::PoolingAggregation::MAX: {
            output = reference.clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); ++i) {
                if (!ensureSameShape(descriptors_per_scale[i])) {
                    output.release();
                    return;
                }
                cv::max(output, descriptors_per_scale[i], output);
            }
            return;
        }
        case thesis_project::PoolingAggregation::MIN: {
            output = reference.clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); ++i) {
                if (!ensureSameShape(descriptors_per_scale[i])) {
                    output.release();
                    return;
                }
                cv::min(output, descriptors_per_scale[i], output);
            }
            return;
        }
        case thesis_project::PoolingAggregation::AVERAGE:
        case thesis_project::PoolingAggregation::WEIGHTED_AVG:
        default:
            break;
    }

    output = cv::Mat::zeros(rows, cols, type);
    double weight_sum = 0.0;

    for (size_t i = 0; i < descriptors_per_scale.size(); ++i) {
        const cv::Mat& mat = descriptors_per_scale[i];
        if (!ensureSameShape(mat)) {
            output.release();
            return;
        }

        double w = 1.0;
        if (aggregation == thesis_project::PoolingAggregation::WEIGHTED_AVG && weights.size() > i) {
            w = weights[i];
        }

        if (w <= 0.0) continue;

        cv::Mat scaled;
        mat.convertTo(scaled, type, w);
        output += scaled;
        weight_sum += w;
    }

    if (weight_sum > 0.0) {
        output.convertTo(output, type, 1.0 / weight_sum);
    } else {
        output = reference.clone();
    }
}

} // namespace thesis_project::pooling
