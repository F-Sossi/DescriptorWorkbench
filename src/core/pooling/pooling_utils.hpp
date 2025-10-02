#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

namespace thesis_project::pooling::utils {

// Normalize each descriptor row to unit norm (L1 or L2)
inline void normalizeRows(cv::Mat& descriptors, int normType) {
    if (descriptors.empty()) return;
    for (int r = 0; r < descriptors.rows; ++r) {
        cv::Mat row = descriptors.row(r);
        cv::normalize(row, row, 1.0, 0.0, normType);
    }
}

// Apply RootSIFT-style element-wise sqrt after L1 normalization
// RootSIFT is designed for histogram-based descriptors (SIFT, HoNC, etc.) which are non-negative.
// For CNN descriptors with negative values, this clamps to zero with a warning.
inline void applyRooting(cv::Mat& descriptors) {
    if (descriptors.empty()) return;

    static bool has_warned_negatives = false;
    bool found_negatives = false;

    for (int i = 0; i < descriptors.rows; ++i) {
        float* ptr = descriptors.ptr<float>(i);
        for (int j = 0; j < descriptors.cols; ++j) {
            float v = ptr[j];
            if (v < 0.0f) {
                if (!has_warned_negatives) {
                    found_negatives = true;
                }
                ptr[j] = 0.0f;  // Clamp negative values to zero
            } else {
                ptr[j] = std::sqrt(v);
            }
        }
    }

    // Warn once about negative values
    if (found_negatives && !has_warned_negatives) {
        std::cerr << "[WARN] RootSIFT applied to descriptor with negative values. "
                  << "Clamping to zero. This is expected for CNN descriptors but not for SIFT/histogram-based descriptors.\n";
        has_warned_negatives = true;
    }
}

} // namespace thesis_project::pooling::utils

