#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace thesis_project::matching {

/**
 * @brief Abstract base class for descriptor matching strategies
 * 
 * This interface defines the contract for different matching algorithms
 * that can be used to find correspondences between descriptors from two images.
 * 
 * The strategy pattern allows for easy swapping of matching algorithms
 * (e.g., BruteForce, FLANN, ratio test, etc.) without changing client code.
 */
class MatchingStrategy {
public:
    virtual ~MatchingStrategy() = default;
    
    /**
     * @brief Match descriptors between two images
     * 
     * @param descriptors1 Descriptors from the first image
     * @param descriptors2 Descriptors from the second image
     * @return std::vector<cv::DMatch> Vector of matches
     */
    virtual std::vector<cv::DMatch> matchDescriptors(
        const cv::Mat& descriptors1, 
        const cv::Mat& descriptors2
    ) = 0;
    
    /**
     * @brief Calculate precision of matches using ground truth
     * 
     * @param matches Vector of descriptor matches
     * @param keypoints2 Keypoints from the second image
     * @param projectedPoints Ground truth projected points from first to second image
     * @param matchThreshold Distance threshold for considering a match correct
     * @return double Precision value (0.0 to 1.0)
     */
    virtual double calculatePrecision(
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::Point2f>& projectedPoints,
        double matchThreshold
    ) = 0;
    
    /**
     * @brief Adjust match threshold based on image scale
     * 
     * @param baseThreshold Base threshold value
     * @param scaleFactor Scale factor of the image
     * @return double Adjusted threshold
     */
    virtual double adjustMatchThreshold(
        double baseThreshold, 
        double scaleFactor
    ) = 0;
    
    /**
     * @brief Get the name of this matching strategy
     * @return std::string Strategy name for logging/debugging
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Check if this strategy supports ratio testing
     * @return bool True if ratio testing is supported
     */
    virtual bool supportsRatioTest() const = 0;
};

using MatchingStrategyPtr = std::unique_ptr<MatchingStrategy>;

} // namespace thesis_project::matching