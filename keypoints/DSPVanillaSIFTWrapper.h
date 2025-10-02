//-------------------------------------------------------------------------
// Name: DSPVanillaSIFTWrapper.h
// Author: Research Team
// Description: DSP-enabled wrapper for VanillaSIFT and derivatives
//              Inherits from VanillaSIFT to access internal pyramid for
//              true pyramid-aware domain size pooling with configurable
//              aggregation methods (average, max, min, concatenate).
//
// This wrapper provides external DSP capabilities that match DSPSIFT's
// performance by reusing the exact same Gaussian pyramid structure.
//-------------------------------------------------------------------------

#ifndef DSP_VANILLA_SIFT_WRAPPER_H
#define DSP_VANILLA_SIFT_WRAPPER_H

#include "VanillaSIFT.h"
#include "RGBSIFT.h"
#include "HoWH.h"
#include "HoNC.h"
#include "include/thesis_project/types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include <opencv2/imgproc.hpp>

/**
 * @brief DSP-enabled wrapper that inherits from VanillaSIFT
 *
 * Template class that wraps any VanillaSIFT-derived descriptor to provide
 * domain size pooling with configurable aggregation. By inheriting from
 * the base SIFT class, we gain access to protected methods like
 * buildGaussianPyramid() and calcSIFTDescriptor(), allowing us to reuse
 * the exact same pyramid structure that SIFT uses internally.
 *
 * Usage:
 *   DSPVanillaSIFTWrapper<VanillaSIFT> dsp_sift;
 *   DSPVanillaSIFTWrapper<RGBSIFT> dsp_rgbsift;
 *   DSPVanillaSIFTWrapper<HoWH> dsp_howh;
 */
namespace thesis_project {

template<typename SiftType = VanillaSIFT>
class DSPVanillaSIFTWrapper : public SiftType {
public:
    // Inherit constructors from base SIFT type
    using SiftType::SiftType;

    /**
     * @brief Compute DSP descriptors using internal pyramid
     * @param image Input image
     * @param keypoints Keypoints to describe
     * @param descriptors Output descriptor matrix
     * @param scales Scale factors for domain pooling (e.g., {0.85, 1.0, 1.30})
     * @param aggregation Aggregation method (AVERAGE, MAX, MIN, CONCATENATE)
     * @param apply_rooting Apply RootSIFT to each scale before pooling
     */
    void computeDSP(const cv::Mat& image,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::Mat& descriptors,
                    const DescriptorParams& params);

private:
    static constexpr bool supportsColorDescriptor() {
        return std::is_base_of_v<cv::RGBSIFT, SiftType> || std::is_same_v<cv::RGBSIFT, SiftType> ||
               std::is_base_of_v<HoWH, SiftType> || std::is_same_v<HoWH, SiftType> ||
               std::is_base_of_v<HoNC, SiftType> || std::is_same_v<HoNC, SiftType>;
    }

    /**
     * @brief Aggregate descriptors from multiple scales
     * @param descriptors_per_scale Vector of descriptor matrices
     * @param output Aggregated output descriptors
     * @param aggregation Aggregation method to use
     */
    void aggregateDescriptors(const std::vector<cv::Mat>& descriptors_per_scale,
                              const DescriptorParams& params,
                              cv::Mat& output,
                              PoolingAggregation aggregation) const;
};

namespace detail {

    inline void l1NormalizeRow(cv::Mat& mat_row) {
        if (cv::norm(mat_row, cv::NORM_L1) > 0.0) {
            cv::normalize(mat_row, mat_row, 1.0, 0.0, cv::NORM_L1);
        } else {
            mat_row.setTo(0.0f);
        }
    }

    inline void applyRoot(cv::Mat& mat_row) {
        float* ptr = mat_row.ptr<float>();
        for (int c = 0; c < mat_row.cols; ++c) {
            float v = ptr[c];
            ptr[c] = (v <= 0.0f) ? 0.0f : std::sqrt(v);
        }
    }

    inline std::vector<float> defaultScales() {
        return {0.85f, 1.0f, 1.30f};
    }

    inline std::vector<float> buildWeights(const std::vector<float>& scales,
                                           const DescriptorParams& params) {
        if (params.pooling_aggregation != PoolingAggregation::WEIGHTED_AVG) {
            return {};
        }

        if (!params.scale_weights.empty() && params.scale_weights.size() == scales.size()) {
            return params.scale_weights;
        }

        std::vector<float> weights(scales.size(), 1.0f);
        switch (params.scale_weighting) {
            case ScaleWeighting::GAUSSIAN: {
                const float sigma = params.scale_weight_sigma > 0.0f ? params.scale_weight_sigma : 0.15f;
                const float center = 1.0f;
                for (size_t i = 0; i < scales.size(); ++i) {
                    const float log_ratio = std::log(scales[i]) - std::log(center);
                    weights[i] = std::exp(-0.5f * (log_ratio * log_ratio) / (sigma * sigma));
                }
                break;
            }
            case ScaleWeighting::TRIANGULAR: {
                const float radius = params.scale_weight_sigma > 0.0f ? params.scale_weight_sigma : 0.15f;
                for (size_t i = 0; i < scales.size(); ++i) {
                    const float dist = std::abs(scales[i] - 1.0f);
                    weights[i] = std::max(0.0f, 1.0f - dist / radius);
                }
                break;
            }
            case ScaleWeighting::UNIFORM:
            default:
                std::fill(weights.begin(), weights.end(), 1.0f);
                break;
        }

        const float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= std::numeric_limits<float>::epsilon()) {
            return std::vector<float>(scales.size(), 1.0f);
        }

        std::vector<float> normalized(weights.size());
        std::transform(weights.begin(), weights.end(), normalized.begin(), [sum](float v) { return v / sum; });
        return normalized;
    }

} // namespace detail

// Template implementation must be in header
template<typename SiftType>
void DSPVanillaSIFTWrapper<SiftType>::computeDSP(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors,
    const DescriptorParams& params) {

    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    std::vector<float> scales = params.scales.empty() ? detail::defaultScales() : params.scales;
    std::vector<float> weights = detail::buildWeights(scales, params);
    const bool apply_root_before = params.rooting_stage == RootingStage::R_BEFORE_POOLING;

    // Prepare base image (color-aware where supported)
    cv::Mat base;
    cv::Mat converted;

    if constexpr (supportsColorDescriptor()) {
        const cv::Mat* source = &image;
        if (image.channels() == 1) {
            cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
            source = &converted;
        }

        base = this->createInitialColorImage(*source, false, static_cast<float>(this->sigma));

        if (base.channels() != 3) {
            cv::cvtColor(base, base, cv::COLOR_GRAY2BGR);
        }
        if (base.depth() != CV_32F) {
            base.convertTo(base, CV_32F);
        }
    } else {
        const cv::Mat* source = &image;
        if (image.channels() > 1) {
            cv::cvtColor(image, converted, cv::COLOR_BGR2GRAY);
            source = &converted;
        }

        base = this->createInitialImage(*source, false, static_cast<float>(this->sigma));

        if (base.channels() != 1) {
            cv::cvtColor(base, base, cv::COLOR_BGR2GRAY);
        }
        if (base.depth() != CV_32F) {
            base.convertTo(base, CV_32F);
        }
    }

    // Build Gaussian pyramid using inherited method (matches descriptor expectations)
    std::vector<cv::Mat> gpyr;
    int nOctaves = cvRound(std::log(static_cast<double>(std::min(base.cols, base.rows))) / std::log(2.0) - 2.0);
    this->buildGaussianPyramid(base, gpyr, nOctaves);

    // Extract first octave info from keypoints to get nOctaveLayers
    int firstOctave = 0;
    int nOctaveLayers = this->nOctaveLayers;

    // Collect descriptors for each scale (following DSPSIFT logic)
    std::vector<cv::Mat> descriptors_per_scale;
    const int descriptor_len = this->descriptorSize();
    const int d = VanillaSIFT::SIFT_DESCR_WIDTH;
    const int n = VanillaSIFT::SIFT_DESCR_HIST_BINS;

    for (float scale_factor : scales) {
        // Allocate descriptor matrix for this scale
        cv::Mat desc(static_cast<int>(keypoints.size()),
                     descriptor_len,
                     CV_32F);
        desc.setTo(0.0f);

        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;

            // Calculate octave/layer for this scale (DSPSIFT logic from lines 160-165)
            float size = static_cast<float>(kpt.size * scale_factor);
            float floatOctave = static_cast<float>(log2(size / (2 * this->sigma)));
            octave = static_cast<int>(floor(floatOctave));
            layer = static_cast<int>(floor((floatOctave - octave) * nOctaveLayers));
            scale = octave >= 0 ? 1.f / (1 << octave) : static_cast<float>(1 << -octave);
            size = size * scale;

            cv::Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);

            // Find pyramid index (DSPSIFT logic from lines 172-174)
            int gpyrLength = static_cast<int>(gpyr.size());
            int wantedIndex = ((octave - firstOctave) * (nOctaveLayers + 3) + layer);
            int correctedIndex = std::max(0, std::min(gpyrLength - 1, wantedIndex));

            // Handle invalid indices (DSPSIFT logic from lines 181-187)
            if (wantedIndex < 0) {
                desc.row(static_cast<int>(i)).setTo(-1.f);
                continue;
            }

            const cv::Mat& img = gpyr[correctedIndex];

            // Calculate angle (DSPSIFT logic from lines 190-192)
            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;

            // Use inherited calcSIFTDescriptor to extract from pyramid level
            this->calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n,
                                     desc.ptr<float>(static_cast<int>(i)));
        }

        // Optional normalization/rooting before pooling
        if (params.normalize_before_pooling || apply_root_before) {
            for (int r = 0; r < desc.rows; ++r) {
                float* ptr = desc.ptr<float>(r);
                bool is_invalid = true;
                for (int c = 0; c < desc.cols; ++c) {
                    if (ptr[c] != -1.f) { is_invalid = false; break; }
                }
                if (is_invalid) {
                    continue;
                }

                cv::Mat row = desc.row(r);
                if (params.normalize_before_pooling) {
                    cv::normalize(row, row, 1.0, 0.0, params.norm_type);
                }
                if (apply_root_before) {
                    detail::l1NormalizeRow(row);
                    detail::applyRoot(row);
                }
            }
        }

        descriptors_per_scale.push_back(desc);
    }

    // Aggregate descriptors using configured method
    aggregateDescriptors(descriptors_per_scale, params, descriptors, params.pooling_aggregation);

    // Root/normalize after pooling if requested
    if (params.rooting_stage == RootingStage::R_AFTER_POOLING) {
        for (int r = 0; r < descriptors.rows; ++r) {
            cv::Mat row = descriptors.row(r);
            detail::l1NormalizeRow(row);
            detail::applyRoot(row);
        }
    }

    if (params.normalize_after_pooling) {
        for (int r = 0; r < descriptors.rows; ++r) {
            cv::Mat row = descriptors.row(r);
            cv::normalize(row, row, 1.0, 0.0, params.norm_type);
        }
    }
}

template<typename SiftType>
void DSPVanillaSIFTWrapper<SiftType>::aggregateDescriptors(
    const std::vector<cv::Mat>& descriptors_per_scale,
    const DescriptorParams& params,
    cv::Mat& output,
    PoolingAggregation aggregation) const {

    if (descriptors_per_scale.empty()) {
        output.release();
        return;
    }

    const int rows = descriptors_per_scale[0].rows;
    const int cols = descriptors_per_scale[0].cols;

    const auto &scales = params.scales.empty() ? detail::defaultScales() : params.scales;
    const std::vector<float> weights = detail::buildWeights(scales, params);

    switch (aggregation) {
        case PoolingAggregation::AVERAGE: {
            output = cv::Mat::zeros(rows, cols, CV_32F);
            std::vector<int> counts(rows * cols, 0);
            for (const auto& desc : descriptors_per_scale) {
                for (int r = 0; r < rows; ++r) {
                    const float* src = desc.ptr<float>(r);
                    float* dst = output.ptr<float>(r);
                    int* cnt = counts.data() + r * cols;
                    for (int c = 0; c < cols; ++c) {
                        float val = src[c];
                        if (val != -1.f) {
                            dst[c] += val;
                            cnt[c] += 1;
                        }
                    }
                }
            }
            for (int r = 0; r < rows; ++r) {
                float* dst = output.ptr<float>(r);
                int* cnt = counts.data() + r * cols;
                for (int c = 0; c < cols; ++c) {
                    if (cnt[c] > 0) {
                        dst[c] /= static_cast<float>(cnt[c]);
                    } else {
                        dst[c] = 0.0f;
                    }
                }
            }
            break;
        }

        case PoolingAggregation::MAX: {
            output = cv::Mat(rows, cols, CV_32F, cv::Scalar(-std::numeric_limits<float>::infinity()));
            for (const auto& desc : descriptors_per_scale) {
                for (int r = 0; r < rows; ++r) {
                    const float* src = desc.ptr<float>(r);
                    float* dst = output.ptr<float>(r);
                    for (int c = 0; c < cols; ++c) {
                        float val = src[c];
                        if (val != -1.f) {
                            dst[c] = std::max(dst[c], val);
                        }
                    }
                }
            }
            output.forEach<float>([](float& v, const int*) {
                if (!std::isfinite(v)) {
                    v = 0.0f;
                }
            });
            break;
        }

        case PoolingAggregation::MIN: {
            output = cv::Mat(rows, cols, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
            for (const auto& desc : descriptors_per_scale) {
                for (int r = 0; r < rows; ++r) {
                    const float* src = desc.ptr<float>(r);
                    float* dst = output.ptr<float>(r);
                    for (int c = 0; c < cols; ++c) {
                        float val = src[c];
                        if (val != -1.f) {
                            dst[c] = std::min(dst[c], val);
                        }
                    }
                }
            }
            output.forEach<float>([](float& v, const int*) {
                if (!std::isfinite(v)) {
                    v = 0.0f;
                }
            });
            break;
        }

        case PoolingAggregation::CONCATENATE: {
            // Horizontal concatenation (increases dimensionality)
            cv::hconcat(descriptors_per_scale, output);
            break;
        }

        case PoolingAggregation::WEIGHTED_AVG:
        default: {
            output = cv::Mat::zeros(rows, cols, CV_32F);
            std::vector<float> weight_sum(rows * cols, 0.0f);
            const std::vector<float> effective_weights = weights.empty() ? std::vector<float>(descriptors_per_scale.size(), 1.0f / static_cast<float>(descriptors_per_scale.size())) : weights;

            for (size_t idx = 0; idx < descriptors_per_scale.size(); ++idx) {
                const float w = (idx < effective_weights.size()) ? effective_weights[idx] : 0.0f;
                if (w <= std::numeric_limits<float>::epsilon()) continue;
                const auto& desc = descriptors_per_scale[idx];
                for (int r = 0; r < rows; ++r) {
                    const float* src = desc.ptr<float>(r);
                    float* dst = output.ptr<float>(r);
                    float* sum_ptr = weight_sum.data() + r * cols;
                    for (int c = 0; c < cols; ++c) {
                        float val = src[c];
                        if (val != -1.f) {
                            dst[c] += w * val;
                            sum_ptr[c] += w;
                        }
                    }
                }
            }

            for (int r = 0; r < rows; ++r) {
                float* dst = output.ptr<float>(r);
                float* sum_ptr = weight_sum.data() + r * cols;
                for (int c = 0; c < cols; ++c) {
                    if (sum_ptr[c] > std::numeric_limits<float>::epsilon()) {
                        dst[c] /= sum_ptr[c];
                    } else {
                        dst[c] = 0.0f;
                    }
                }
            }
            break;
        }
    }
}

} // namespace thesis_project

#endif // DSP_VANILLA_SIFT_WRAPPER_H
