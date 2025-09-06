#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>

namespace thesis_project::pooling::utils {

// Normalize each descriptor row to unit norm (L1 or L2)
inline void normalizeRows(cv::Mat& descriptors, int normType) {
    if (descriptors.empty()) return;
    for (int r = 0; r < descriptors.rows; ++r) {
        cv::Mat row = descriptors.row(r);
        cv::normalize(row, row, 1.0, 0.0, normType);
    }
}

// Apply RootSIFT-style element-wise sqrt after L1 normalization (assumes non-negative values)
inline void applyRooting(cv::Mat& descriptors) {
    if (descriptors.empty()) return;
    for (int i = 0; i < descriptors.rows; ++i) {
        float* ptr = descriptors.ptr<float>(i);
        for (int j = 0; j < descriptors.cols; ++j) {
            float v = ptr[j];
            ptr[j] = (v >= 0.0f) ? std::sqrt(v) : -std::sqrt(-v);
        }
    }
}

} // namespace thesis_project::pooling::utils

