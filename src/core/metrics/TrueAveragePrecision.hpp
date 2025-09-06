#ifndef CORE_METRICS_TRUE_AVERAGE_PRECISION_HPP
#define CORE_METRICS_TRUE_AVERAGE_PRECISION_HPP

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <opencv2/core.hpp>

/**
 * @brief Utilities for computing true Information Retrieval style Mean Average Precision (mAP)
 * 
 * This implements proper IR-style mAP computation using:
 * - Ground truth relevance via homography projection  
 * - Ranked descriptor matching results
 * - Single-GT policy (R=1) with pixel tolerance
 * 
 * Based on HPatches evaluation methodology.
 */
namespace TrueAveragePrecision {

    /**
     * @brief Simple 2D point structure
     */
    struct Point2D {
        double x, y;
        Point2D(double x = 0.0, double y = 0.0) : x(x), y(y) {}
        Point2D(const cv::Point2f& p) : x(p.x), y(p.y) {}
        Point2D(const cv::KeyPoint& kp) : x(kp.pt.x), y(kp.pt.y) {}
    };

    /**
     * @brief Result of AP computation for a single query
     */
    struct QueryAPResult {
        double ap = 0.0;                  // Average Precision for this query
        int rank_of_true_match = -1;      // 1-based rank of true match (-1 if no relevant item found)
        int total_relevant = 0;           // Total relevant items (R=1 for single-GT policy)
        bool has_potential_match = false; // Whether query had any potential ground truth match
    };

    /**
     * @brief Project point using homography matrix
     * @param H Homography matrix in row-major format [h00,h01,h02, h10,h11,h12, h20,h21,h22]
     * @param p Point to project
     * @return Projected point (may be infinite if singular)
     */
    Point2D projectPoint(const std::array<double, 9>& H, const Point2D& p);

    /**
     * @brief Convert OpenCV Mat homography to array format
     * @param H_mat OpenCV 3x3 homography matrix (CV_64F)
     * @return Homography in row-major array format
     */
    std::array<double, 9> matToArray(const cv::Mat& H_mat);

    /**
     * @brief Find single most relevant keypoint index (R=1 policy)
     * 
     * Projects query keypoint from image A to image B using homography,
     * then finds the closest keypoint in B within pixel tolerance.
     * 
     * @param queryA Query keypoint in image A
     * @param H_A_to_B Homography mapping A→B
     * @param keypointsB All detected keypoints in image B  
     * @param tau_px Pixel tolerance for geometric correctness (typically 3.0)
     * @return Index of relevant keypoint in B, or -1 if none within tolerance
     */
    int findSingleRelevantIndex(const Point2D& queryA,
                               const std::array<double, 9>& H_A_to_B,
                               const std::vector<Point2D>& keypointsB,
                               double tau_px = 3.0);

    /**
     * @brief Compute Average Precision from ranked relevance labels
     * 
     * Standard IR-style AP computation:
     * AP = (1/R) * Σ(Precision@k * 1[rel[k]=1])
     * 
     * @param relevance_ranked Binary relevance in ranked order (top-1, top-2, ...)
     * @return Average Precision value [0.0, 1.0]
     */
    double computeAveragePrecision(const std::vector<int>& relevance_ranked);

    /**
     * @brief Compute AP for a single query using descriptor distances
     * 
     * Full pipeline:
     * 1. Determine ground truth relevance via homography projection
     * 2. Rank all candidates by descriptor distance
     * 3. Build relevance vector in ranked order
     * 4. Compute Average Precision
     * 
     * @param queryA Query keypoint in image A
     * @param H_A_to_B Homography mapping A→B  
     * @param keypointsB All detected keypoints in image B
     * @param distances_to_B Descriptor distances from query to each B keypoint (same order as keypointsB)
     * @param tau_px Pixel tolerance for geometric correctness (typically 3.0)
     * @return QueryAPResult containing AP and diagnostic info
     */
    QueryAPResult computeQueryAP(const Point2D& queryA,
                                const std::array<double, 9>& H_A_to_B,
                                const std::vector<Point2D>& keypointsB,
                                const std::vector<double>& distances_to_B,
                                double tau_px = 3.0);

    /**
     * @brief Convenience wrapper using OpenCV types
     */
    QueryAPResult computeQueryAP(const cv::KeyPoint& queryA,
                                const cv::Mat& H_A_to_B,
                                const std::vector<cv::KeyPoint>& keypointsB,
                                const std::vector<double>& distances_to_B,
                                double tau_px = 3.0);

    /**
     * @brief Distance calculation utilities
     */
    inline double euclideanDistance(const Point2D& a, const Point2D& b) {
        return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

} // namespace TrueAveragePrecision

#endif // CORE_METRICS_TRUE_AVERAGE_PRECISION_HPP