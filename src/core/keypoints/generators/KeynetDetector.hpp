#pragma once

#include "src/interfaces/IKeypointGenerator.hpp"
#include "thesis_project/types.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace thesis_project {

/**
 * @brief KeyNet detector using Kornia Python bridge
 *
 * This detector uses Kornia's KeyNet implementation through a Python bridge.
 * KeyNet is a learned keypoint detector that combines handcrafted and CNN filters
 * and is designed to work well with CNN descriptors like HardNet and SOSNet.
 *
 * The detector calls a Python script that uses Kornia to extract keypoints,
 * then reads the results back into C++. This approach allows us to leverage
 * the state-of-the-art KeyNet implementation while maintaining integration
 * with our existing C++ pipeline.
 */
class KeynetDetector : public IKeypointGenerator {
public:
    /**
     * @brief Constructor for KeyNet detector
     * @param max_keypoints Maximum number of keypoints to detect (default: 2000)
     * @param device Device to use for computation ("auto", "cuda", "cpu")
     */
    explicit KeynetDetector(int max_keypoints = 2000, const std::string& device = "auto");

    /**
     * @brief Detect keypoints in an image using KeyNet
     * @param image Input image (grayscale or color)
     * @param params Detection parameters (max_features will be used)
     * @return Vector of detected keypoints
     */
    std::vector<cv::KeyPoint> detect(
        const cv::Mat& image,
        const KeypointParams& params = {}
    ) override;

    /**
     * @brief Detect keypoints with spatial non-overlap constraint using KeyNet
     * @param image Input image
     * @param min_distance Minimum euclidean distance between keypoint centers
     * @param params Detection parameters
     * @return Vector of non-overlapping keypoints
     */
    std::vector<cv::KeyPoint> detectNonOverlapping(
        const cv::Mat& image,
        float min_distance,
        const KeypointParams& params = {}
    ) override;

    /**
     * @brief Get human-readable detector name
     * @return Detector name string
     */
    std::string name() const override {
        return "KeyNet";
    }

    /**
     * @brief Get detector type enum
     * @return KeypointGenerator enum value
     */
    KeypointGenerator type() const override {
        return KeypointGenerator::KEYNET;
    }

    /**
     * @brief Check if Python/Kornia environment is available
     * @return true if KeyNet can be used, false otherwise
     */
    static bool isAvailable();

private:
    int max_keypoints_;
    std::string device_;
    std::string python_script_path_;

    /**
     * @brief Save image to temporary file for Python processing
     * @param image Image to save
     * @param temp_path Output path for temporary image
     */
    void saveImageToTemp(const cv::Mat& image, const std::string& temp_path) const;

    /**
     * @brief Load keypoints from temporary CSV file written by Python
     * @param csv_path Path to CSV file with keypoint data
     * @return Vector of keypoints loaded from CSV
     */
    std::vector<cv::KeyPoint> loadKeypointsFromTemp(const std::string& csv_path) const;

    /**
     * @brief Execute Python KeyNet detection script
     * @param image_path Path to input image
     * @param output_csv_path Path where Python will write keypoints
     * @return true if Python execution succeeded, false otherwise
     */
    bool executePythonKeyNet(const std::string& image_path, const std::string& output_csv_path) const;

    /**
     * @brief Generate unique temporary filename
     * @param extension File extension (with dot, e.g., ".png")
     * @return Unique temporary file path
     */
    std::string generateTempFilename(const std::string& extension) const;
};

} // namespace thesis_project