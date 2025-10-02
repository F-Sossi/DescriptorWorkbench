#include "locked_in_keypoints.hpp"
#include "../processing/processor_utils.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace fs = boost::filesystem;

constexpr int BORDER = 35;  // HPatches compliance: 65px patches need ~35px margin
constexpr int DEFAULT_MAX_KEYPOINTS = 2000;

void LockedInKeypoints::generateLockedInKeypoints(const std::string& dataFolderPath, const std::string& referenceKeypointsBaseFolder) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();
            const fs::path referenceKeypointsFolder = fs::path(referenceKeypointsBaseFolder) / subfolderName;
            fs::create_directories(referenceKeypointsFolder);

            const fs::path image1Path = subfolderPath / "1.ppm";
            cv::Mat image1 = cv::imread(image1Path.generic_string(), cv::IMREAD_COLOR);
            if (image1.empty()) {
                throw std::runtime_error("Failed to read image: " + image1Path.generic_string());
            }

            cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
            std::vector<cv::KeyPoint> keypoints;
            sift->detect(image1, keypoints);

            // Filter keypoints that are too close to the image border
            keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                return keypoint.pt.x < static_cast<float>(BORDER) || keypoint.pt.y < static_cast<float>(BORDER) ||
                       keypoint.pt.x > static_cast<float>(image1.cols - BORDER) || keypoint.pt.y > static_cast<float>(image1.rows - BORDER);
            }), keypoints.end());

            // If no keypoints are left after filtering, skip this series
            if (keypoints.empty()) {
                std::cerr << "All keypoints are too close to the border for folder: " << subfolderName << std::endl;
                continue;
            }

            std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response > b.response;
            });

            if (keypoints.size() > static_cast<size_t>(DEFAULT_MAX_KEYPOINTS)) {
                keypoints.resize(DEFAULT_MAX_KEYPOINTS);
            }

            // Print the number of keypoints and the folder name
            std::cout << "Number of keypoints: " << keypoints.size() << " Folder name: " << subfolderName << std::endl;

            const fs::path keypointsFile1 = referenceKeypointsFolder / "1ppm.csv";
            saveKeypointsToCSV(keypointsFile1.generic_string(), keypoints);

            for (int i = 2; i <= 6; ++i) {
                const fs::path homographyFile = subfolderPath / ("H_1_" + std::to_string(i));
                cv::Mat homography = readHomography(homographyFile.generic_string());

                std::vector<cv::Point2f> points;
                for (const auto& keypoint : keypoints) {
                    points.push_back(keypoint.pt);
                }

                std::vector<cv::Point2f> transformedPoints;
                cv::perspectiveTransform(points, transformedPoints, homography);

                std::vector<cv::KeyPoint> transformedKeypoints;
                for (size_t j = 0; j < transformedPoints.size(); ++j) {
                    const cv::KeyPoint& originalKeypoint = keypoints[j];
                    const cv::Point2f& transformedPoint = transformedPoints[j];
                    transformedKeypoints.emplace_back(transformedPoint, originalKeypoint.size, originalKeypoint.angle,
                                                      originalKeypoint.response, originalKeypoint.octave, originalKeypoint.class_id);
                }

                // Filter transformed keypoints that are too close to the image border
                transformedKeypoints.erase(std::remove_if(transformedKeypoints.begin(), transformedKeypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                    return keypoint.pt.x < static_cast<float>(BORDER) || keypoint.pt.y < static_cast<float>(BORDER) ||
                           keypoint.pt.x > static_cast<float>(image1.cols - BORDER) || keypoint.pt.y > static_cast<float>(image1.rows - BORDER);
                }), transformedKeypoints.end());

                // If no transformed keypoints are left after filtering, skip this file
                if (transformedKeypoints.empty()) {
                    std::cerr << "All transformed keypoints are too close to the border for file: " << i << " in folder: " << subfolderName << std::endl;
                    continue;
                }

                const fs::path keypointsFile = referenceKeypointsFolder / (std::to_string(i) + "ppm.csv");
                saveKeypointsToCSV(keypointsFile.generic_string(), transformedKeypoints);
            }
        }
    }
}

void LockedInKeypoints::generateLockedInKeypointsToDatabase(const std::string& dataFolderPath,
                                                           const thesis_project::database::DatabaseManager& db,
                                                           int max_keypoints) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();

            const fs::path image1Path = subfolderPath / "1.ppm";
            cv::Mat image1 = cv::imread(image1Path.generic_string(), cv::IMREAD_COLOR);
            if (image1.empty()) {
                throw std::runtime_error("Failed to read image: " + image1Path.generic_string());
            }

            cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();
            std::vector<cv::KeyPoint> keypoints;
            sift->detect(image1, keypoints);

            // Filter keypoints that are too close to the image border
            keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                return keypoint.pt.x < static_cast<float>(BORDER) || keypoint.pt.y < static_cast<float>(BORDER) ||
                       keypoint.pt.x > static_cast<float>(image1.cols - BORDER) || keypoint.pt.y > static_cast<float>(image1.rows - BORDER);
            }), keypoints.end());

            // If no keypoints are left after filtering, skip this series
            if (keypoints.empty()) {
                std::cerr << "All keypoints are too close to the border for folder: " << subfolderName << std::endl;
                continue;
            }

            std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response > b.response;
            });

            if (keypoints.size() > static_cast<size_t>(DEFAULT_MAX_KEYPOINTS)) {
                keypoints.resize(DEFAULT_MAX_KEYPOINTS);
            }

            // Print the number of keypoints and the folder name
            std::cout << "Number of keypoints: " << keypoints.size() << " Folder name: " << subfolderName << std::endl;

            // Store keypoints for image 1 in database
            if (!db.storeLockedKeypoints(subfolderName, "1.ppm", keypoints)) {
                std::cerr << "Failed to store keypoints for " << subfolderName << "/1.ppm" << std::endl;
                continue;
            }

            for (int i = 2; i <= 6; ++i) {
                const fs::path homographyFile = subfolderPath / ("H_1_" + std::to_string(i));
                cv::Mat homography = readHomography(homographyFile.generic_string());

                std::vector<cv::Point2f> points;
                for (const auto& keypoint : keypoints) {
                    points.push_back(keypoint.pt);
                }

                std::vector<cv::Point2f> transformedPoints;
                cv::perspectiveTransform(points, transformedPoints, homography);

                std::vector<cv::KeyPoint> transformedKeypoints;
                for (size_t j = 0; j < transformedPoints.size(); ++j) {
                    const cv::KeyPoint& originalKeypoint = keypoints[j];
                    const cv::Point2f& transformedPoint = transformedPoints[j];
                    transformedKeypoints.emplace_back(transformedPoint, originalKeypoint.size, originalKeypoint.angle,
                                                      originalKeypoint.response, originalKeypoint.octave, originalKeypoint.class_id);
                }

                // Filter transformed keypoints that are too close to the image border
                transformedKeypoints.erase(std::remove_if(transformedKeypoints.begin(), transformedKeypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                    return keypoint.pt.x < static_cast<float>(BORDER) || keypoint.pt.y < static_cast<float>(BORDER) ||
                           keypoint.pt.x > static_cast<float>(image1.cols - BORDER) || keypoint.pt.y > static_cast<float>(image1.rows - BORDER);
                }), transformedKeypoints.end());

                // If no transformed keypoints are left after filtering, skip this file
                if (transformedKeypoints.empty()) {
                    std::cerr << "All transformed keypoints are too close to the border for file: " << i << " in folder: " << subfolderName << std::endl;
                    continue;
                }

                // Store transformed keypoints in database
                std::string imageName = std::to_string(i) + ".ppm";
                if (!db.storeLockedKeypoints(subfolderName, imageName, transformedKeypoints)) {
                    std::cerr << "Failed to store keypoints for " << subfolderName << "/" << imageName << std::endl;
                }
            }
        }
    }
}

void LockedInKeypoints::generateLockedInKeypointsToDatabase(const std::string& dataFolderPath,
                                                           const thesis_project::database::DatabaseManager& db,
                                                           int keypoint_set_id,
                                                           int max_keypoints) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    int sift_features = max_keypoints > 0 ? max_keypoints : 0;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(sift_features);
    const int BORDER = 40;

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();
            std::cout << "Processing scene: " << subfolderName << std::endl;

            // Process the reference image (1.ppm) first
            fs::path imagePath = subfolderPath / "1.ppm";
            if (!fs::exists(imagePath)) {
                std::cerr << "Reference image 1.ppm not found in folder: " << subfolderName << std::endl;
                continue;
            }

            cv::Mat image1 = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
            if (image1.empty()) {
                std::cerr << "Could not load image: " << imagePath.string() << std::endl;
                continue;
            }

            std::vector<cv::KeyPoint> keypoints;
            detector->detect(image1, keypoints);

            // Apply boundary filtering to original keypoints
            keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                return keypoint.pt.x < BORDER || keypoint.pt.y < BORDER ||
                       keypoint.pt.x > (image1.cols - BORDER) || keypoint.pt.y > (image1.rows - BORDER);
            }), keypoints.end());

            if (max_keypoints > 0 && keypoints.size() > static_cast<size_t>(max_keypoints)) {
                std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                    return a.response > b.response;
                });
                keypoints.resize(max_keypoints);
            }

            std::cout << "Number of keypoints: " << keypoints.size() << " Folder name: " << subfolderName << std::endl;

            // Store reference image keypoints
            if (db.storeLockedKeypointsForSet(keypoint_set_id, subfolderName, "1.ppm", keypoints)) {
                std::cout << "Stored " << keypoints.size() << " keypoints for " << subfolderName << "/1.ppm" << std::endl;
            } else {
                std::cerr << "Failed to store keypoints for " << subfolderName << "/1.ppm" << std::endl;
            }

            // Process other images (2.ppm to 6.ppm)
            for (int i = 2; i <= 6; ++i) {
                cv::Mat homography = processor_utils::readHomography((subfolderPath / ("H_1_" + std::to_string(i))).string());
                if (homography.empty()) {
                    std::cerr << "Could not load homography file for image " << i << " in folder: " << subfolderName << std::endl;
                    continue;
                }

                // Transform keypoints using homography
                std::vector<cv::KeyPoint> transformedKeypoints = keypoints;
                std::vector<cv::Point2f> originalPoints, transformedPoints;
                cv::KeyPoint::convert(keypoints, originalPoints);
                cv::perspectiveTransform(originalPoints, transformedPoints, homography);

                for (size_t j = 0; j < transformedKeypoints.size(); ++j) {
                    transformedKeypoints[j].pt = transformedPoints[j];
                }

                // Apply boundary filtering to transformed keypoints
                transformedKeypoints.erase(std::remove_if(transformedKeypoints.begin(), transformedKeypoints.end(), [image1](const cv::KeyPoint& keypoint) {
                    return keypoint.pt.x < BORDER || keypoint.pt.y < BORDER ||
                           keypoint.pt.x > (image1.cols - BORDER) || keypoint.pt.y > (image1.rows - BORDER);
                }), transformedKeypoints.end());

                if (transformedKeypoints.empty()) {
                    std::cerr << "All transformed keypoints are too close to the border for file: " << i << " in folder: " << subfolderName << std::endl;
                    continue;
                }

                // Store transformed keypoints using keypoint set
                std::string imageName = std::to_string(i) + ".ppm";
                if (db.storeLockedKeypointsForSet(keypoint_set_id, subfolderName, imageName, transformedKeypoints)) {
                    std::cout << "Stored " << transformedKeypoints.size() << " keypoints for " << subfolderName << "/" << imageName << std::endl;
                } else {
                    std::cerr << "Failed to store keypoints for " << subfolderName << "/" << imageName << std::endl;
                }
            }
        }
    }
}

void LockedInKeypoints::saveKeypointsToCSV(const std::string& filePath, const std::vector<cv::KeyPoint>& keypoints) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filePath);
    }

    file << "x,y,size,angle,response,octave,class_id\n";
    for (const auto& keypoint : keypoints) {
        file << keypoint.pt.x << "," << keypoint.pt.y << "," << keypoint.size << "," << keypoint.angle << ","
             << keypoint.response << "," << keypoint.octave << "," << keypoint.class_id << "\n";
    }

    file.close();
}

cv::Mat LockedInKeypoints::readHomography(const std::string& filePath) {
    cv::Mat homography(3, 3, CV_64F);
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filePath);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            file >> homography.at<double>(i, j);
        }
    }

    file.close();
    return homography;
}

void LockedInKeypoints::displayLockedInKeypoints(const std::string& dataFolderPath) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    // Initialize database for keypoint loading
    thesis_project::database::DatabaseConfig db_config;
    db_config.connection_string = "experiments.db";
    db_config.enabled = true;
    thesis_project::database::DatabaseManager db(db_config);

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();

            // Create a vector to store the image filenames
            std::vector<std::string> imageFilenames;
            for (int i = 1; i <= 6; i++) {
                fs::path imageFilename = subfolderPath / (std::to_string(i) + ".ppm");
                imageFilenames.push_back(imageFilename.generic_string());
            }

            // Create a vector to store the image matrices and keypoints
            std::vector<std::pair<cv::Mat, std::vector<cv::KeyPoint>>> imagesWithKeypoints;

            // Process each image and store the image matrix and keypoints
            for (size_t i = 0; i < imageFilenames.size(); i++) {
                const std::string& imageFilename = imageFilenames[i];
                cv::Mat image = cv::imread(imageFilename, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cerr << "Failed to read image: " << imageFilename << std::endl;
                    continue;
                }

                // Load keypoints from database instead of CSV
                std::string imageName = std::to_string(i + 1) + ".ppm";
                std::vector<cv::KeyPoint> keypoints = db.getLockedKeypoints(subfolderName, imageName);
                
                if (keypoints.empty()) {
                    std::cerr << "No keypoints found in database for " << subfolderName << "/" << imageName << std::endl;
                    continue;
                }
                imagesWithKeypoints.emplace_back(image, keypoints);
            }

            // Display each image with keypoints in a separate window
            for (size_t i = 0; i < imagesWithKeypoints.size(); i++) {
                const auto& imageWithKeypoints = imagesWithKeypoints[i];
                const cv::Mat& image = imageWithKeypoints.first;
                const std::vector<cv::KeyPoint>& keypoints = imageWithKeypoints.second;

                cv::Mat imageWithKeypointsDisplay = image.clone();
                cv::drawKeypoints(image, keypoints, imageWithKeypointsDisplay, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                std::string windowName = subfolderName + " - Image " + std::to_string(i + 1) + " - Locked-In Keypoints";
                cv::imshow(windowName, imageWithKeypointsDisplay);
            }

            cv::waitKey(0); // Wait for a key press to continue

            // Close all the displayed image windows
            for (size_t i = 0; i < imagesWithKeypoints.size(); i++) {
                std::string windowName = subfolderName + " - Image " + std::to_string(i + 1) + " - Locked-In Keypoints";
                cv::destroyWindow(windowName);
            }
        }
    }
}

void LockedInKeypoints::displayLockedInKeypointsBorder(const std::string& dataFolderPath) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    // Initialize database for keypoint loading
    thesis_project::database::DatabaseConfig db_config;
    db_config.connection_string = "experiments.db";
    db_config.enabled = true;
    thesis_project::database::DatabaseManager db(db_config);

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();

            // Create a vector to store the image filenames
            std::vector<std::string> imageFilenames;
            for (int i = 1; i <= 6; i++) {
                fs::path imageFilename = subfolderPath / (std::to_string(i) + ".ppm");
                imageFilenames.push_back(imageFilename.generic_string());
            }

            // Create a vector to store the image matrices and keypoints
            std::vector<std::pair<cv::Mat, std::vector<cv::KeyPoint>>> imagesWithKeypoints;

            // Process each image and store the image matrix and keypoints
            for (size_t i = 0; i < imageFilenames.size(); i++) {
                const std::string& imageFilename = imageFilenames[i];
                cv::Mat image = cv::imread(imageFilename, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cerr << "Failed to read image: " << imageFilename << std::endl;
                    continue;
                }

                // Load keypoints from database instead of CSV
                std::string imageName = std::to_string(i + 1) + ".ppm";
                std::vector<cv::KeyPoint> keypoints = db.getLockedKeypoints(subfolderName, imageName);
                
                if (keypoints.empty()) {
                    std::cerr << "No keypoints found in database for " << subfolderName << "/" << imageName << std::endl;
                    continue;
                }
                imagesWithKeypoints.emplace_back(image, keypoints);
            }

            // Display each image with keypoints in a separate window
            for (size_t i = 0; i < imagesWithKeypoints.size(); i++) {
                const auto& imageWithKeypoints = imagesWithKeypoints[i];
                const cv::Mat& image = imageWithKeypoints.first;
                const std::vector<cv::KeyPoint>& keypoints = imageWithKeypoints.second;

                cv::Mat imageWithKeypointsDisplay = image.clone();
                cv::drawKeypoints(image, keypoints, imageWithKeypointsDisplay, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // Draw a 65x65 pixel square centered on each keypoint
                for (const auto& keypoint : keypoints) {
                    cv::Point2f pt = keypoint.pt;
                    cv::rectangle(imageWithKeypointsDisplay,
                                  cv::Point(pt.x - 32.5, pt.y - 32.5),
                                  cv::Point(pt.x + 32.5, pt.y + 32.5),
                                  cv::Scalar(0, 255, 0), 2);
                }

                std::string windowName = subfolderName + " - Image " + std::to_string(i + 1) + " - Locked-In Keypoints";
                cv::imshow(windowName, imageWithKeypointsDisplay);
            }

            cv::waitKey(0); // Wait for a key press to continue

            // Close all the displayed image windows
            for (size_t i = 0; i < imagesWithKeypoints.size(); i++) {
                std::string windowName = subfolderName + " - Image " + std::to_string(i + 1) + " - Locked-In Keypoints";
                cv::destroyWindow(windowName);
            }
        }
    }
}


std::vector<cv::KeyPoint> LockedInKeypoints::readKeypointsFromCSV(const std::string& filePath) {
    std::vector<cv::KeyPoint> keypoints;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filePath);
    }

    std::string line;
    std::getline(file, line); // Skip the header line

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        cv::KeyPoint keypoint;
        std::getline(ss, value, ',');
        keypoint.pt.x = std::stof(value);
        std::getline(ss, value, ',');
        keypoint.pt.y = std::stof(value);
        std::getline(ss, value, ',');
        keypoint.size = std::stof(value);
        std::getline(ss, value, ',');
        keypoint.angle = std::stof(value);
        std::getline(ss, value, ',');
        keypoint.response = std::stof(value);
        std::getline(ss, value, ',');
        keypoint.octave = std::stoi(value);
        std::getline(ss, value);
        keypoint.class_id = std::stoi(value);

        keypoints.push_back(keypoint);
    }

    file.close();
    return keypoints;
}

void LockedInKeypoints::saveLockedInKeypointsBorder(const std::string& dataFolderPath) {
    if (!fs::exists(dataFolderPath) || !fs::is_directory(dataFolderPath)) {
        throw std::runtime_error("Invalid data folder path: " + dataFolderPath);
    }

    // Initialize database for keypoint loading
    thesis_project::database::DatabaseConfig db_config;
    db_config.connection_string = "experiments.db";
    db_config.enabled = true;
    thesis_project::database::DatabaseManager db(db_config);

    // Create output directory for saved images
    std::string outputPath = "keypoint_visualizations/";
    fs::create_directories(outputPath);
    std::cout << "\nSaving keypoint visualizations to: " << fs::absolute(outputPath).string() << std::endl;

    for (const auto& entry : fs::directory_iterator(dataFolderPath)) {
        const fs::path subfolderPath = entry.path();
        if (fs::is_directory(subfolderPath)) {
            const std::string subfolderName = subfolderPath.filename().generic_string();

            // Check if keypoints exist in database
            auto availableImages = db.getAvailableImages(subfolderName);
            if (availableImages.empty()) {
                std::cerr << "No keypoints found in database for: " << subfolderName << std::endl;
                continue;
            }

            // Create subfolder for this image set
            std::string setOutputPath = outputPath + subfolderName + "/";
            fs::create_directories(setOutputPath);
            std::cout << "\nProcessing: " << subfolderName << std::endl;

            // Create a vector to store the image filenames
            std::vector<std::string> imageFilenames;
            for (int i = 1; i <= 6; i++) {
                fs::path imageFilename = subfolderPath / (std::to_string(i) + ".ppm");
                imageFilenames.push_back(imageFilename.generic_string());
            }

            // Process each image and save with keypoints
            for (size_t i = 0; i < imageFilenames.size(); i++) {
                const std::string& imageFilename = imageFilenames[i];
                cv::Mat image = cv::imread(imageFilename, cv::IMREAD_COLOR);

                if (image.empty()) {
                    std::cerr << "Failed to read image: " << imageFilename << std::endl;
                    continue;
                }

                // Load keypoints from database
                std::string imageName = std::to_string(i + 1) + ".ppm";
                std::vector<cv::KeyPoint> keypoints = db.getLockedKeypoints(subfolderName, imageName);
                
                if (keypoints.empty()) {
                    std::cerr << "No keypoints found in database for " << subfolderName << "/" << imageName << std::endl;
                    continue;
                }
                cv::Mat imageWithKeypointsDisplay = image.clone();

                // Draw keypoints
                cv::drawKeypoints(image, keypoints, imageWithKeypointsDisplay,
                                  cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // Draw a 65x65 pixel square centered on each keypoint
                for (const auto& keypoint : keypoints) {
                    cv::Point2f pt = keypoint.pt;
                    cv::rectangle(imageWithKeypointsDisplay,
                                  cv::Point(pt.x - 32.5, pt.y - 32.5),
                                  cv::Point(pt.x + 32.5, pt.y + 32.5),
                                  cv::Scalar(0, 255, 0), 2);
                }

                // Add text with keypoint count and image info
                std::string text1 = "Image " + std::to_string(i + 1) + " - Keypoints: " + std::to_string(keypoints.size());
                std::string text2 = subfolderName;
                cv::putText(imageWithKeypointsDisplay, text1, cv::Point(10, 30),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                cv::putText(imageWithKeypointsDisplay, text2, cv::Point(10, 65),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

                // Save the image
                std::string outputFilename = setOutputPath + "image_" + std::to_string(i + 1) + "_keypoints.jpg";
                cv::imwrite(outputFilename, imageWithKeypointsDisplay);
                std::cout << "  Saved: image_" << std::to_string(i + 1) << "_keypoints.jpg ("
                         << keypoints.size() << " keypoints)" << std::endl;
            }

            // Create a summary image showing all 6 images in a grid
            createSummaryImageWithDatabase(imageFilenames, subfolderName, setOutputPath, db);
        }
    }

    std::cout << "\n=== All visualizations saved ===" << std::endl;
    std::cout << "Location: " << fs::absolute(outputPath).string() << std::endl;
    std::cout << "You can view them on your host system." << std::endl;
}

// Add this helper function for creating the summary image
void LockedInKeypoints::createSummaryImage(const std::vector<std::string>& imageFilenames,
                                           const std::string& sceneName,
                                           const std::string& outputPath) {
    // CSV fallback implementation - legacy support
    const fs::path referenceKeypointsFolder = fs::path(KEYPOINTS_PATH) / sceneName;
    // Create a 2x3 grid for 6 images
    int gridCols = 3;
    int gridRows = 2;
    int thumbWidth = 400;
    int thumbHeight = 300;
    int borderSize = 5;

    // Create white background with borders
    cv::Mat summary((thumbHeight + borderSize) * gridRows + borderSize,
                    (thumbWidth + borderSize) * gridCols + borderSize,
                    CV_8UC3, cv::Scalar(255, 255, 255));

    for (size_t i = 0; i < imageFilenames.size() && i < 6; i++) {
        cv::Mat image = cv::imread(imageFilenames[i], cv::IMREAD_COLOR);
        if (image.empty()) continue;

        // Load keypoints from CSV (fallback)
        fs::path keypointsFilename = referenceKeypointsFolder / (std::to_string(i + 1) + "ppm.csv");
        if (!fs::exists(keypointsFilename)) continue;

        std::vector<cv::KeyPoint> keypoints = readKeypointsFromCSV(keypointsFilename.generic_string());

        // Draw keypoints on image
        cv::Mat imageWithKeypoints;
        cv::drawKeypoints(image, keypoints, imageWithKeypoints,
                         cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Resize to thumbnail
        cv::Mat thumbnail;
        cv::resize(imageWithKeypoints, thumbnail, cv::Size(thumbWidth, thumbHeight));

        // Add image number and keypoint count
        std::string text = "Image " + std::to_string(i + 1) + " (" + std::to_string(keypoints.size()) + " kpts)";
        cv::putText(thumbnail, text, cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // Calculate position in grid with borders
        int row = i / gridCols;
        int col = i % gridCols;
        int x = col * (thumbWidth + borderSize) + borderSize;
        int y = row * (thumbHeight + borderSize) + borderSize;

        // Copy to summary image
        cv::Rect roi(x, y, thumbWidth, thumbHeight);
        thumbnail.copyTo(summary(roi));
    }

    // Add title to the summary
    cv::putText(summary, "Scene: " + sceneName, cv::Point(20, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 2);

    // Save summary
    std::string summaryFilename = outputPath + "summary_all_images.jpg";
    cv::imwrite(summaryFilename, summary);
    std::cout << "  Saved: summary_all_images.jpg" << std::endl;
}

// Database-enabled version of createSummaryImage
void LockedInKeypoints::createSummaryImageWithDatabase(const std::vector<std::string>& imageFilenames,
                                                      const std::string& sceneName,
                                                      const std::string& outputPath,
                                                      const thesis_project::database::DatabaseManager& db) {
    // Create a 2x3 grid for 6 images
    int gridCols = 3;
    int gridRows = 2;
    int thumbWidth = 400;
    int thumbHeight = 300;
    int borderSize = 5;

    // Create white background with borders
    cv::Mat summary((thumbHeight + borderSize) * gridRows + borderSize,
                    (thumbWidth + borderSize) * gridCols + borderSize,
                    CV_8UC3, cv::Scalar(255, 255, 255));

    for (size_t i = 0; i < imageFilenames.size() && i < 6; i++) {
        cv::Mat image = cv::imread(imageFilenames[i], cv::IMREAD_COLOR);
        if (image.empty()) continue;

        // Load keypoints from database
        std::string imageName = std::to_string(i + 1) + ".ppm";
        std::vector<cv::KeyPoint> keypoints = db.getLockedKeypoints(sceneName, imageName);
        if (keypoints.empty()) continue;

        // Draw keypoints on image
        cv::Mat imageWithKeypoints;
        cv::drawKeypoints(image, keypoints, imageWithKeypoints,
                         cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Resize to thumbnail
        cv::Mat thumbnail;
        cv::resize(imageWithKeypoints, thumbnail, cv::Size(thumbWidth, thumbHeight));

        // Add image number and keypoint count
        std::string text = "Image " + std::to_string(i + 1) + " (" + std::to_string(keypoints.size()) + " kpts)";
        cv::putText(thumbnail, text, cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // Calculate position in grid with borders
        int row = i / gridCols;
        int col = i % gridCols;
        int x = col * (thumbWidth + borderSize) + borderSize;
        int y = row * (thumbHeight + borderSize) + borderSize;

        // Copy thumbnail to summary at calculated position
        cv::Rect roi(x, y, thumbWidth, thumbHeight);
        thumbnail.copyTo(summary(roi));
    }

    // Add title to the summary
    cv::putText(summary, "Scene: " + sceneName + " (DB)", cv::Point(20, 30),
               cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 0), 2);

    // Save summary
    std::string summaryFilename = outputPath + "summary_all_images.jpg";
    cv::imwrite(summaryFilename, summary);
    std::cout << "  Saved: summary_all_images.jpg (from database)" << std::endl;
}
