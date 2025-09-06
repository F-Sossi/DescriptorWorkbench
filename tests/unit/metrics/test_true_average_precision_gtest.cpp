#include <gtest/gtest.h>
#include "src/core/metrics/TrueAveragePrecision.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>

class TrueAveragePrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Identity homography matrix (no transformation)
        identity_H = {1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0};
        
        // Translation homography matrix (shift by 10, 5)
        translation_H = {1.0, 0.0, 10.0,
                          0.0, 1.0,  5.0,
                          0.0, 0.0,  1.0};
        
        // Sample keypoints for testing
        keypoints_A = {
            TrueAveragePrecision::Point2D(100.0, 200.0),
            TrueAveragePrecision::Point2D(150.0, 250.0),
            TrueAveragePrecision::Point2D(200.0, 300.0)
        };
        
        keypoints_B = {
            TrueAveragePrecision::Point2D(110.0, 205.0),  // Close to transformed point (110, 205)
            TrueAveragePrecision::Point2D(160.0, 255.0),  // Close to transformed point (160, 255)
            TrueAveragePrecision::Point2D(500.0, 500.0),  // Far away point
            TrueAveragePrecision::Point2D(210.0, 305.0)   // Close to transformed point (210, 305)
        };
    }

    std::array<double, 9> identity_H;
    std::array<double, 9> translation_H;
    std::vector<TrueAveragePrecision::Point2D> keypoints_A;
    std::vector<TrueAveragePrecision::Point2D> keypoints_B;
};

TEST_F(TrueAveragePrecisionTest, ProjectPointIdentity) {
    TrueAveragePrecision::Point2D p(100.0, 200.0);
    TrueAveragePrecision::Point2D result = TrueAveragePrecision::projectPoint(identity_H, p);
    
    EXPECT_DOUBLE_EQ(result.x, 100.0);
    EXPECT_DOUBLE_EQ(result.y, 200.0);
}

TEST_F(TrueAveragePrecisionTest, ProjectPointTranslation) {
    TrueAveragePrecision::Point2D p(100.0, 200.0);
    TrueAveragePrecision::Point2D result = TrueAveragePrecision::projectPoint(translation_H, p);
    
    EXPECT_DOUBLE_EQ(result.x, 110.0);  // 100 + 10
    EXPECT_DOUBLE_EQ(result.y, 205.0);  // 200 + 5
}

TEST_F(TrueAveragePrecisionTest, ProjectPointOrigin) {
    TrueAveragePrecision::Point2D origin(0.0, 0.0);
    TrueAveragePrecision::Point2D result = TrueAveragePrecision::projectPoint(translation_H, origin);
    
    EXPECT_DOUBLE_EQ(result.x, 10.0);
    EXPECT_DOUBLE_EQ(result.y, 5.0);
}

TEST_F(TrueAveragePrecisionTest, MatToArrayConversion) {
    // Create OpenCV homography matrix
    cv::Mat H_mat = (cv::Mat_<double>(3, 3) << 
        2.0, 0.5, 10.0,
        0.0, 1.5, 15.0,
        0.0, 0.0,  1.0);
    
    std::array<double, 9> result = TrueAveragePrecision::matToArray(H_mat);
    
    // Check row-major order conversion
    EXPECT_DOUBLE_EQ(result[0], 2.0);   // H(0,0)
    EXPECT_DOUBLE_EQ(result[1], 0.5);   // H(0,1)
    EXPECT_DOUBLE_EQ(result[2], 10.0);  // H(0,2)
    EXPECT_DOUBLE_EQ(result[3], 0.0);   // H(1,0)
    EXPECT_DOUBLE_EQ(result[4], 1.5);   // H(1,1)
    EXPECT_DOUBLE_EQ(result[5], 15.0);  // H(1,2)
    EXPECT_DOUBLE_EQ(result[6], 0.0);   // H(2,0)
    EXPECT_DOUBLE_EQ(result[7], 0.0);   // H(2,1)
    EXPECT_DOUBLE_EQ(result[8], 1.0);   // H(2,2)
}

TEST_F(TrueAveragePrecisionTest, FindSingleRelevantIndexExactMatch) {
    // With translation homography, point (100, 200) maps to (110, 205)
    // keypoints_B[0] is at (110, 205) - exact match
    TrueAveragePrecision::Point2D query(100.0, 200.0);
    
    int result = TrueAveragePrecision::findSingleRelevantIndex(query, translation_H, keypoints_B, 3.0);
    
    EXPECT_EQ(result, 0);  // First keypoint in B should match
}

TEST_F(TrueAveragePrecisionTest, FindSingleRelevantIndexWithinTolerance) {
    // Point (150, 250) maps to (160, 255)
    // keypoints_B[1] is at (160, 255) - within tolerance
    TrueAveragePrecision::Point2D query(150.0, 250.0);
    
    int result = TrueAveragePrecision::findSingleRelevantIndex(query, translation_H, keypoints_B, 5.0);
    
    EXPECT_EQ(result, 1);  // Second keypoint should match
}

TEST_F(TrueAveragePrecisionTest, FindSingleRelevantIndexOutsideTolerance) {
    // Use a point that doesn't map close to any keypoint in B
    TrueAveragePrecision::Point2D query(50.0, 50.0);  // Maps to (60, 55)
    
    int result = TrueAveragePrecision::findSingleRelevantIndex(query, translation_H, keypoints_B, 3.0);
    
    EXPECT_EQ(result, -1);  // No match within tolerance
}

TEST_F(TrueAveragePrecisionTest, FindSingleRelevantIndexEmptyKeypoints) {
    std::vector<TrueAveragePrecision::Point2D> empty_keypoints;
    TrueAveragePrecision::Point2D query(100.0, 200.0);
    
    int result = TrueAveragePrecision::findSingleRelevantIndex(query, translation_H, empty_keypoints, 3.0);
    
    EXPECT_EQ(result, -1);  // No keypoints to match
}

TEST_F(TrueAveragePrecisionTest, ComputeAveragePrecisionPerfectRanking) {
    // All relevant items are ranked first
    std::vector<int> relevance = {1, 1, 0, 0, 0};  // 2 relevant items in top 2 positions
    
    double ap = TrueAveragePrecision::computeAveragePrecision(relevance);
    
    // AP = (1/2) * (1*1 + 1*1) = 1.0
    EXPECT_DOUBLE_EQ(ap, 1.0);
}

TEST_F(TrueAveragePrecisionTest, ComputeAveragePrecisionWorstRanking) {
    // All relevant items are ranked last
    std::vector<int> relevance = {0, 0, 0, 1, 1};  // 2 relevant items in last 2 positions
    
    double ap = TrueAveragePrecision::computeAveragePrecision(relevance);
    
    // AP = (1/2) * (P@4 * 1 + P@5 * 1) = (1/2) * (1/4 + 2/5) = (1/2) * (0.25 + 0.4) = 0.325
    EXPECT_DOUBLE_EQ(ap, 0.325);
}

TEST_F(TrueAveragePrecisionTest, ComputeAveragePrecisionNoRelevant) {
    // No relevant items
    std::vector<int> relevance = {0, 0, 0, 0, 0};
    
    double ap = TrueAveragePrecision::computeAveragePrecision(relevance);
    
    EXPECT_DOUBLE_EQ(ap, 0.0);
}

TEST_F(TrueAveragePrecisionTest, ComputeAveragePrecisionSingleRelevant) {
    // Single relevant item at different positions
    std::vector<int> relevance_rank1 = {1, 0, 0, 0, 0};  // Rank 1
    std::vector<int> relevance_rank3 = {0, 0, 1, 0, 0};  // Rank 3
    std::vector<int> relevance_rank5 = {0, 0, 0, 0, 1};  // Rank 5
    
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::computeAveragePrecision(relevance_rank1), 1.0);      // P@1 = 1/1
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::computeAveragePrecision(relevance_rank3), 1.0/3.0); // P@3 = 1/3
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::computeAveragePrecision(relevance_rank5), 1.0/5.0); // P@5 = 1/5
}

TEST_F(TrueAveragePrecisionTest, ComputeAveragePrecisionEmpty) {
    std::vector<int> empty_relevance;
    
    double ap = TrueAveragePrecision::computeAveragePrecision(empty_relevance);
    
    EXPECT_DOUBLE_EQ(ap, 0.0);
}

TEST_F(TrueAveragePrecisionTest, ComputeQueryAPWithMatch) {
    TrueAveragePrecision::Point2D query(100.0, 200.0);
    
    // Distances: close to relevant keypoint, farther from others
    std::vector<double> distances = {0.5, 5.0, 100.0, 10.0};  // Closest is index 0 (relevant)
    
    auto result = TrueAveragePrecision::computeQueryAP(query, translation_H, keypoints_B, distances, 3.0);
    
    EXPECT_TRUE(result.has_potential_match);
    EXPECT_EQ(result.rank_of_true_match, 1);  // Best ranked (lowest distance)
    EXPECT_EQ(result.total_relevant, 1);
    EXPECT_DOUBLE_EQ(result.ap, 1.0);  // Perfect ranking
}

TEST_F(TrueAveragePrecisionTest, ComputeQueryAPNoMatch) {
    // Query that doesn't map close to any keypoint in B
    TrueAveragePrecision::Point2D query(50.0, 50.0);  
    
    std::vector<double> distances = {0.5, 5.0, 100.0, 10.0};
    
    auto result = TrueAveragePrecision::computeQueryAP(query, translation_H, keypoints_B, distances, 3.0);
    
    EXPECT_FALSE(result.has_potential_match);
    EXPECT_EQ(result.rank_of_true_match, -1);
    EXPECT_EQ(result.total_relevant, 0);
    EXPECT_DOUBLE_EQ(result.ap, 0.0);
}

TEST_F(TrueAveragePrecisionTest, ComputeQueryAPWorstRank) {
    TrueAveragePrecision::Point2D query(100.0, 200.0);
    
    // Make the relevant match have the highest (worst) distance
    std::vector<double> distances = {100.0, 0.5, 1.0, 2.0};  // Relevant match (index 0) has worst distance
    
    auto result = TrueAveragePrecision::computeQueryAP(query, translation_H, keypoints_B, distances, 3.0);
    
    EXPECT_TRUE(result.has_potential_match);
    EXPECT_EQ(result.rank_of_true_match, 4);  // Worst rank (4th position)
    EXPECT_EQ(result.total_relevant, 1);
    EXPECT_DOUBLE_EQ(result.ap, 0.25);  // AP = P@4 = 1/4
}

TEST_F(TrueAveragePrecisionTest, OpenCVConvenienceWrapper) {
    // Test the OpenCV wrapper function
    cv::KeyPoint query_kp(100.0, 200.0, 1.0);
    cv::Mat H_mat = (cv::Mat_<double>(3, 3) << 
        1.0, 0.0, 10.0,
        0.0, 1.0,  5.0,
        0.0, 0.0,  1.0);
    
    std::vector<cv::KeyPoint> cv_keypoints_B;
    for (const auto& p : keypoints_B) {
        cv_keypoints_B.emplace_back(p.x, p.y, 1.0);
    }
    
    std::vector<double> distances = {0.5, 5.0, 100.0, 10.0};
    
    auto result = TrueAveragePrecision::computeQueryAP(query_kp, H_mat, cv_keypoints_B, distances, 3.0);
    
    EXPECT_TRUE(result.has_potential_match);
    EXPECT_EQ(result.rank_of_true_match, 1);
    EXPECT_DOUBLE_EQ(result.ap, 1.0);
}

TEST_F(TrueAveragePrecisionTest, EuclideanDistanceCalculation) {
    TrueAveragePrecision::Point2D p1(0.0, 0.0);
    TrueAveragePrecision::Point2D p2(3.0, 4.0);
    TrueAveragePrecision::Point2D p3(1.0, 1.0);
    
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::euclideanDistance(p1, p2), 5.0);        // 3-4-5 triangle
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::euclideanDistance(p1, p3), std::sqrt(2.0)); // Diagonal
    EXPECT_DOUBLE_EQ(TrueAveragePrecision::euclideanDistance(p1, p1), 0.0);        // Same point
}

// Parameterized test for precision@k scenarios  
class AveragePrecisionParameterizedTest : public ::testing::TestWithParam<std::tuple<std::vector<int>, double>> {};

TEST_P(AveragePrecisionParameterizedTest, ComputeAveragePrecisionVariousCases) {
    std::vector<int> relevance = std::get<0>(GetParam());
    double expected_ap = std::get<1>(GetParam());
    
    double result = TrueAveragePrecision::computeAveragePrecision(relevance);
    EXPECT_NEAR(result, expected_ap, 1e-6);  // Use NEAR for floating point comparison
}

INSTANTIATE_TEST_SUITE_P(
    APTests,
    AveragePrecisionParameterizedTest,
    ::testing::Values(
        // Perfect ranking cases
        std::make_tuple(std::vector<int>{1}, 1.0),
        std::make_tuple(std::vector<int>{1, 1}, 1.0),
        std::make_tuple(std::vector<int>{1, 0, 1}, (1.0 + 2.0/3.0)/2.0),  // AP = (1/2)*(1 + 2/3) = 5/6
        
        // Mixed cases
        std::make_tuple(std::vector<int>{0, 1, 0, 1}, (0.5 + 0.5)/2.0),   // AP = (1/2)*(1/2 + 2/4) = 0.5
        std::make_tuple(std::vector<int>{1, 0, 0, 1}, (1.0 + 0.5)/2.0),   // AP = (1/2)*(1 + 2/4) = 0.75
        
        // Edge cases
        std::make_tuple(std::vector<int>{0, 0, 0}, 0.0),  // No relevant
        std::make_tuple(std::vector<int>{1, 0, 0}, 1.0)   // Single relevant at top
    )
);