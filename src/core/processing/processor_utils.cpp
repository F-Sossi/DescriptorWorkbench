#include "processor_utils.hpp"
#include "../config/experiment_config.hpp"
#include "keypoints/VanillaSIFT.h"
#include "src/core/pooling/PoolingFactory.hpp"
#include "src/core/matching/MatchingFactory.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp> // For cv::Ptr and core functionalities

// Assume baseWidth and baseHeight represent the dimensions of your base resolution
double baseWidth = 800.0; // Example base width
double baseHeight = 600.0; // Example base height

double processor_utils::calculateRelativeScalingFactor(const cv::Mat& image) {
    double scaleFactorWidth = image.cols / baseWidth;
    double scaleFactorHeight = image.rows / baseHeight;
    // You can use either width or height or a combination based on your application's needs
    // Here we take the average to accommodate both dimensions
    double averageScaleFactor = (scaleFactorWidth + scaleFactorHeight) / 2.0;
    return averageScaleFactor;
}

double processor_utils::adjustMatchThresholdForImageSet(double baseThreshold, double scaleFactor) {
    // Use default brute force matching strategy for threshold adjustment
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(BRUTE_FORCE);
    return matchingStrategy->adjustMatchThreshold(baseThreshold, scaleFactor);
}

cv::Mat processor_utils::applyGaussianNoise(const cv::Mat& image, double mean, double stddev) {
    cv::Mat noisyImage = image.clone();
    cv::Mat noise(image.size(), image.type());

    cv::randn(noise, mean, stddev); // Generate Gaussian noise
    noisyImage += noise; // Add the noise to the original image

    return noisyImage;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> processor_utils::detectAndCompute(const cv::Ptr<cv::Feature2D>& detector, const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    return std::make_pair(keypoints, descriptors);
}

void processor_utils::sumPoolingDetectAndCompute(const cv::Ptr<cv::Feature2D>& detector,
                                       const cv::Mat& image,
                                       std::vector<cv::KeyPoint>& keypoints,
                                       cv::Mat& descriptors) {
}

void processor_utils::rootDescriptors(cv::Mat& descriptors) {
    for (int i = 0; i < descriptors.rows; ++i) {
        for (int j = 0; j < descriptors.cols; ++j) {
            descriptors.at<float>(i, j) = std::sqrt(descriptors.at<float>(i, j));
        }
    }
}

std::vector<cv::DMatch> processor_utils::matchDescriptors(const cv::Mat& descriptors1, const cv::Mat& descriptors2, MatchingStrategy strategy) {
    // Use the specified matching strategy
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(strategy);
    return matchingStrategy->matchDescriptors(descriptors1, descriptors2);
}

double processor_utils::calculatePrecision(const std::vector<cv::DMatch>& matches,
                                           const std::vector<cv::KeyPoint>& keypoints2,
                                           const std::vector<cv::Point2f>& projectedPoints,
                                           double matchThreshold) {
    // Use default brute force matching strategy for precision calculation
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(BRUTE_FORCE);
    return matchingStrategy->calculatePrecision(matches, keypoints2, projectedPoints, matchThreshold);
}

void processor_utils::saveResults(const std::string& filePath, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& dataRows) {
    std::ofstream file;

    // Check if the file already exists
    const bool fileExists = std::ifstream(filePath).good();

    // Open the file in append mode
    file.open(filePath, std::ios::out | std::ios::app);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    // Write headers only if the file doesn't exist
    if (!fileExists) {
        for (std::size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << ",";
        }
        file << "\n";
    }

    // Write data rows
    for (const auto& row : dataRows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

void processor_utils::saveKeypointsToCSV(const std::string& filePath, const std::vector<cv::KeyPoint>& keypoints) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    // Write header
    file << "x,y,size,angle,response,octave,class_id" << std::endl;

    // Write keypoints
    for (const auto& kp : keypoints) {
        file << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << "," << kp.response << "," << kp.octave << "," << kp.class_id << std::endl;
    }

    file.close();
}

void processor_utils::saveDescriptorsToCSV(const std::string& filePath, const cv::Mat& descriptors) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    // Write descriptors
    for (int row = 0; row < descriptors.rows; ++row) {
        std::stringstream ss;
        for (int col = 0; col < descriptors.cols; ++col) {
            ss << descriptors.at<float>(row, col);
            if (col < descriptors.cols - 1) {
                ss << ",";
            }
        }
        file << ss.str() << std::endl;
    }

    file.close();
}

cv::Mat processor_utils::readHomography(const std::string& filePath) {
    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Unable to open the homography file: " << filePath << "\n";
        exit(-1);
    }

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            file >> H.at<double>(row, col);
        }
    }

    file.close();
    return H;
}



std::pair<std::vector<cv::KeyPoint>, cv::Mat> processor_utils::detectAndComputeWithConfigLocked(const cv::Mat& image, const std::vector<cv::KeyPoint>& lockedKeypoints, const experiment_config &config) {
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> result;

    if (config.descriptorOptions.UseLockedInKeypoints) {
        // Use the provided locked-in keypoints
        result.first = lockedKeypoints;
    } else {
        // Detect keypoints using the specified detector
        result = processor_utils::detectAndCompute(config.detector, image);
    }

    // New interface routing (default): for supported descriptors and strategies with extractor overloads
    if (thesis_project::factories::DescriptorFactory::isSupported(config) &&
        (config.descriptorOptions.poolingStrategy == NONE ||
         config.descriptorOptions.poolingStrategy == DOMAIN_SIZE_POOLING ||
         config.descriptorOptions.poolingStrategy == STACKING)) {
        try {
            auto extractor = thesis_project::factories::DescriptorFactory::create(config);
            auto poolingStrategy = thesis_project::pooling::PoolingFactory::createFromConfig(config);
            result.second = poolingStrategy->computeDescriptors(image, result.first, *extractor, config);
            return result;
        } catch (...) {
            // Fall through to legacy path on any error
        }
    }

    // Legacy path using detector interface
    auto poolingStrategy = thesis_project::pooling::PoolingFactory::createFromConfig(config);
    result.second = poolingStrategy->computeDescriptors(image, result.first, config.detector, config);

    return result;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> processor_utils::detectAndComputeWithConfig(const cv::Mat& image, const experiment_config &config) {
    // New interface routing (default): for supported descriptors and strategies with extractor overloads
    if (thesis_project::factories::DescriptorFactory::isSupported(config) &&
        (config.descriptorOptions.poolingStrategy == NONE ||
         config.descriptorOptions.poolingStrategy == DOMAIN_SIZE_POOLING ||
         config.descriptorOptions.poolingStrategy == STACKING)) {
        try {
            // Detect keypoints using the configured detector first
            auto detected = processor_utils::detectAndCompute(config.detector, image);
            auto poolingStrategy = thesis_project::pooling::PoolingFactory::createFromConfig(config);
            auto extractor = thesis_project::factories::DescriptorFactory::create(config);
            cv::Mat desc = poolingStrategy->computeDescriptors(image, detected.first, *extractor, config);
            return {detected.first, desc};
        } catch (...) {
            // Fall through to legacy path on any error
        }
    }

    std::pair<std::vector<cv::KeyPoint>, cv::Mat> result;
    // Detect keypoints first
    result = processor_utils::detectAndCompute(config.detector, image);
    
    // Use the new pooling strategy system
    auto poolingStrategy = thesis_project::pooling::PoolingFactory::createFromConfig(config);
    result.second = poolingStrategy->computeDescriptors(image, result.first, config.detector, config);

    return result;
}
