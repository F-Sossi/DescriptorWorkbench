#include "processor_utils.hpp"
#include "src/core/matching/MatchingFactory.hpp"
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
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(
        thesis_project::MatchingMethod::BRUTE_FORCE);
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

std::vector<cv::DMatch> processor_utils::matchDescriptors(const cv::Mat& descriptors1,
                                                         const cv::Mat& descriptors2,
                                                         thesis_project::MatchingMethod method) {
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(method);
    return matchingStrategy->matchDescriptors(descriptors1, descriptors2);
}

double processor_utils::calculatePrecision(const std::vector<cv::DMatch>& matches,
                                           const std::vector<cv::KeyPoint>& keypoints2,
                                           const std::vector<cv::Point2f>& projectedPoints,
                                           double matchThreshold) {
    auto matchingStrategy = thesis_project::matching::MatchingFactory::createStrategy(
        thesis_project::MatchingMethod::BRUTE_FORCE);
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
