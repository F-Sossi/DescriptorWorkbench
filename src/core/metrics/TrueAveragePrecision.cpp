#include "TrueAveragePrecision.hpp"

namespace TrueAveragePrecision {

Point2D projectPoint(const std::array<double, 9>& H, const Point2D& p) {
    const double X = H[0] * p.x + H[1] * p.y + H[2];
    const double Y = H[3] * p.x + H[4] * p.y + H[5];
    const double Z = H[6] * p.x + H[7] * p.y + H[8];
    
    if (std::abs(Z) < 1e-12) {
        return Point2D(std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity());
    }
    return Point2D(X / Z, Y / Z);
}

std::array<double, 9> matToArray(const cv::Mat& H_mat) {
    std::array<double, 9> H;
    CV_Assert(H_mat.rows == 3 && H_mat.cols == 3);
    CV_Assert(H_mat.type() == CV_64F || H_mat.type() == CV_32F);
    
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (H_mat.type() == CV_64F) {
                H[i * 3 + j] = H_mat.at<double>(i, j);
            } else {
                H[i * 3 + j] = static_cast<double>(H_mat.at<float>(i, j));
            }
        }
    }
    return H;
}

int findSingleRelevantIndex(const Point2D& queryA,
                           const std::array<double, 9>& H_A_to_B,
                           const std::vector<Point2D>& keypointsB,
                           double tau_px) {
    const Point2D projected = projectPoint(H_A_to_B, queryA);
    
    // Check if projection is valid
    if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
        return -1;
    }
    
    // Optional: Early-out for projections far outside reasonable image bounds
    // (Avoids wasted nearest-neighbor search for extreme wide baselines)
    constexpr double MAX_IMAGE_BOUND = 2000.0; // Reasonable for most datasets
    if (projected.x < -MAX_IMAGE_BOUND || projected.x > MAX_IMAGE_BOUND ||
        projected.y < -MAX_IMAGE_BOUND || projected.y > MAX_IMAGE_BOUND) {
        return -1;
    }

    int best_idx = -1;
    double best_distance = std::numeric_limits<double>::infinity();
    
    for (int j = 0; j < static_cast<int>(keypointsB.size()); ++j) {
        double distance = euclideanDistance(projected, keypointsB[j]);
        if (distance < best_distance) {
            best_distance = distance;
            best_idx = j;
        }
    }
    
    return (best_distance <= tau_px) ? best_idx : -1;
}

double computeAveragePrecision(const std::vector<int>& relevance_ranked) {
    int total_relevant = std::accumulate(relevance_ranked.begin(), relevance_ranked.end(), 0);
    if (total_relevant <= 0) {
        return 0.0;
    }

    double ap_sum = 0.0;
    int hits = 0;
    
    for (int k = 0; k < static_cast<int>(relevance_ranked.size()); ++k) {
        if (relevance_ranked[k]) {
            hits += 1;
            double precision_at_k = static_cast<double>(hits) / static_cast<double>(k + 1);
            ap_sum += precision_at_k;
        }
    }
    
    return ap_sum / static_cast<double>(total_relevant);
}

QueryAPResult computeQueryAP(const Point2D& queryA,
                            const std::array<double, 9>& H_A_to_B,
                            const std::vector<Point2D>& keypointsB,
                            const std::vector<double>& distances_to_B,
                            double tau_px) {
    QueryAPResult result;

    // Find ground truth relevant keypoint
    const int gt_idx = findSingleRelevantIndex(queryA, H_A_to_B, keypointsB, tau_px);
    
    if (gt_idx == -1) {
        // No relevant item found - this query has R=0
        result.ap = 0.0;
        result.rank_of_true_match = -1;
        result.total_relevant = 0;
        result.has_potential_match = false;
        return result;
    }

    result.has_potential_match = true;
    result.total_relevant = 1; // Single-GT policy

    // Optimized O(N) ranking for R=1 case (no full sort needed)
    const double gt_distance = distances_to_B[gt_idx];
    
    // Count how many items rank better than ground truth
    int better_count = 0;
    int tie_count = 0; // For tie-breaking (items with same distance)
    
    for (int i = 0; i < static_cast<int>(distances_to_B.size()); ++i) {
        if (i == gt_idx) continue; // Skip the ground truth itself
        
        if (distances_to_B[i] < gt_distance) {
            better_count++;
        } else if (distances_to_B[i] == gt_distance) {
            tie_count++; // Count ties for proper ranking
        }
    }
    
    // Rank calculation with tie-breaking (use average rank for ties)
    int rank_1_based = 1 + better_count + (tie_count + 1) / 2;
    
    // For R=1, AP = 1.0 / rank
    result.ap = 1.0 / static_cast<double>(rank_1_based);
    result.rank_of_true_match = rank_1_based;
    
    return result;
}

QueryAPResult computeQueryAP(const cv::KeyPoint& queryA,
                            const cv::Mat& H_A_to_B,
                            const std::vector<cv::KeyPoint>& keypointsB,
                            const std::vector<double>& distances_to_B,
                            double tau_px) {
    // Convert OpenCV types to internal representation
    Point2D query_pt(queryA);
    std::array<double, 9> H_array = matToArray(H_A_to_B);
    
    std::vector<Point2D> keypoints_B_pts;
    keypoints_B_pts.reserve(keypointsB.size());
    for (const auto& kp : keypointsB) {
        keypoints_B_pts.emplace_back(kp);
    }
    
    return computeQueryAP(query_pt, H_array, keypoints_B_pts, distances_to_B, tau_px);
}

} // namespace TrueAveragePrecision