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
#include "include/thesis_project/types.hpp"

using namespace cv;
using namespace thesis_project;

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
     */
    void computeDSP(const Mat& image,
                    std::vector<KeyPoint>& keypoints,
                    Mat& descriptors,
                    const std::vector<float>& scales,
                    PoolingAggregation aggregation = PoolingAggregation::MAX);

private:
    /**
     * @brief Aggregate descriptors from multiple scales
     * @param descriptors_per_scale Vector of descriptor matrices
     * @param output Aggregated output descriptors
     * @param aggregation Aggregation method to use
     */
    void aggregateDescriptors(const std::vector<Mat>& descriptors_per_scale,
                              Mat& output,
                              PoolingAggregation aggregation) const;
};

// Template implementation must be in header
template<typename SiftType>
void DSPVanillaSIFTWrapper<SiftType>::computeDSP(
    const Mat& image,
    std::vector<KeyPoint>& keypoints,
    Mat& descriptors,
    const std::vector<float>& scales,
    PoolingAggregation aggregation) {

    if (keypoints.empty()) {
        descriptors = Mat();
        return;
    }

    // Use inherited createInitialImage to prepare base (matches SIFT's preprocessing)
    Mat base = this->createInitialImage(image, false, static_cast<float>(this->sigma));

    // Build Gaussian pyramid using inherited method (exact same as SIFT uses)
    std::vector<Mat> gpyr;
    int nOctaves = cvRound(log(static_cast<double>(std::min(base.cols, base.rows))) / log(2.) - 2);
    this->buildGaussianPyramid(base, gpyr, nOctaves);

    // Extract first octave info from keypoints to get nOctaveLayers
    int firstOctave = 0;
    int nOctaveLayers = this->nOctaveLayers;

    // Collect descriptors for each scale (following DSPSIFT logic)
    std::vector<Mat> descriptors_per_scale;
    const int d = VanillaSIFT::SIFT_DESCR_WIDTH;
    const int n = VanillaSIFT::SIFT_DESCR_HIST_BINS;

    for (float scale_factor : scales) {
        // Allocate descriptor matrix for this scale
        Mat desc(static_cast<int>(keypoints.size()),
                 d * d * n,  // SIFT descriptor size: 4*4*8 = 128
                 CV_32F);

        for (size_t i = 0; i < keypoints.size(); i++) {
            KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;

            // Calculate octave/layer for this scale (DSPSIFT logic from lines 160-165)
            float size = static_cast<float>(kpt.size * scale_factor);
            float floatOctave = static_cast<float>(log2(size / (2 * this->sigma)));
            octave = static_cast<int>(floor(floatOctave));
            layer = static_cast<int>(floor((floatOctave - octave) * nOctaveLayers));
            scale = octave >= 0 ? 1.f / (1 << octave) : static_cast<float>(1 << -octave);
            size = size * scale;

            Point2f ptf(kpt.pt.x * scale, kpt.pt.y * scale);

            // Find pyramid index (DSPSIFT logic from lines 172-174)
            int gpyrLength = static_cast<int>(gpyr.size());
            int wantedIndex = ((octave - firstOctave) * (nOctaveLayers + 3) + layer);
            int correctedIndex = std::max(0, std::min(gpyrLength - 1, wantedIndex));

            // Handle invalid indices (DSPSIFT logic from lines 181-187)
            if (wantedIndex < 0) {
                for (int descC = 0; descC < desc.cols; descC++) {
                    desc.at<float>(static_cast<int>(i), descC) = -1.f;
                }
                continue;
            }

            const Mat& img = gpyr[correctedIndex];

            // Calculate angle (DSPSIFT logic from lines 190-192)
            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;

            // Use inherited calcSIFTDescriptor to extract from pyramid level
            this->calcSIFTDescriptor(img, ptf, angle, size * 0.5f, d, n,
                                    desc.ptr<float>(static_cast<int>(i)));
        }

        descriptors_per_scale.push_back(desc);
    }

    // Aggregate descriptors using configured method
    aggregateDescriptors(descriptors_per_scale, descriptors, aggregation);
}

template<typename SiftType>
void DSPVanillaSIFTWrapper<SiftType>::aggregateDescriptors(
    const std::vector<Mat>& descriptors_per_scale,
    Mat& output,
    PoolingAggregation aggregation) const {

    if (descriptors_per_scale.empty()) {
        output = Mat();
        return;
    }

    const int rows = descriptors_per_scale[0].rows;
    const int cols = descriptors_per_scale[0].cols;

    switch (aggregation) {
        case PoolingAggregation::AVERAGE: {
            // Average pooling (DSPSIFT default - lines 199-220)
            output = Mat::zeros(rows, cols, CV_32F);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int validScales = 0;
                    float sum = 0.0f;
                    for (const auto& desc : descriptors_per_scale) {
                        float val = desc.at<float>(r, c);
                        if (val != -1.f) {
                            sum += val;
                            validScales++;
                        }
                    }
                    output.at<float>(r, c) = (validScales > 0) ? (sum / validScales) : 0.0f;
                }
            }
            break;
        }

        case PoolingAggregation::MAX: {
            // Max pooling (element-wise maximum)
            output = descriptors_per_scale[0].clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); i++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        float val = descriptors_per_scale[i].at<float>(r, c);
                        if (val != -1.f) {
                            output.at<float>(r, c) = std::max(output.at<float>(r, c), val);
                        }
                    }
                }
            }
            break;
        }

        case PoolingAggregation::MIN: {
            // Min pooling (element-wise minimum)
            output = descriptors_per_scale[0].clone();
            for (size_t i = 1; i < descriptors_per_scale.size(); i++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < cols; c++) {
                        float val = descriptors_per_scale[i].at<float>(r, c);
                        if (val != -1.f) {
                            output.at<float>(r, c) = std::min(output.at<float>(r, c), val);
                        }
                    }
                }
            }
            break;
        }

        case PoolingAggregation::CONCATENATE: {
            // Horizontal concatenation (increases dimensionality)
            hconcat(descriptors_per_scale, output);
            break;
        }

        case PoolingAggregation::WEIGHTED_AVG:
        default: {
            // Default to average
            output = Mat::zeros(rows, cols, CV_32F);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int validScales = 0;
                    float sum = 0.0f;
                    for (const auto& desc : descriptors_per_scale) {
                        float val = desc.at<float>(r, c);
                        if (val != -1.f) {
                            sum += val;
                            validScales++;
                        }
                    }
                    output.at<float>(r, c) = (validScales > 0) ? (sum / validScales) : 0.0f;
                }
            }
            break;
        }
    }
}

#endif // DSP_VANILLA_SIFT_WRAPPER_H
