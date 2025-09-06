#include "VisualVerification.hpp"
#include "experiment_config.hpp"
#include "processor_utils.hpp"
#include "src/core/pooling/PoolingFactory.hpp"
#include <iostream>
#include <algorithm>

cv::Scalar VisualVerification::getKeypointColor(size_t index) {
    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),     // Blue
        cv::Scalar(0, 255, 0),     // Green
        cv::Scalar(0, 0, 255),     // Red
        cv::Scalar(255, 255, 0),   // Cyan
        cv::Scalar(255, 0, 255),   // Magenta
        cv::Scalar(0, 255, 255),   // Yellow
        // Add more colors if needed
    };
    return colors[index % colors.size()];
}

void VisualVerification::verifyMatches(const std::string& folder, 
                                      const std::string& results_folder, 
                                      const experiment_config& config) {
    std::cout << "Processing folder: " << folder << "\nResults folder: " << results_folder << std::endl;

    // Read in 1.ppm as the base image
    cv::Mat image1 = cv::imread(folder + "/1.ppm", cv::IMREAD_COLOR);
    if (config.descriptorOptions.descriptorColorSpace == D_BW) {
        cv::cvtColor(image1, image1, cv::COLOR_BGR2RGB);
    }

    // Use detector from config to detect keypoints in image 1 using the processor utils detect and compute function
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> result1 = processor_utils::detectAndCompute(config.detector, image1);
    std::vector<cv::KeyPoint> keypoints1 = result1.first;
    cv::Mat descriptors1 = result1.second;

    // Compute DSP descriptors for image 1 using factory system
    auto dspPoolingStrategy = thesis_project::pooling::PoolingFactory::createStrategy(DOMAIN_SIZE_POOLING);
    cv::Mat dspDescriptors1 = dspPoolingStrategy->computeDescriptors(image1, keypoints1, config.detector, config);

    for (int i = 2; i <= 6; i++) {
        std::string imagePath = folder + "/" + std::to_string(i) + ".ppm";
        cv::Mat image2 = cv::imread(imagePath, cv::IMREAD_COLOR);

        if (image2.empty()) continue;
        if (config.descriptorOptions.descriptorColorSpace == D_BW) {
            cv::cvtColor(image2, image2, cv::COLOR_BGR2RGB);
        }

        // Detect and compute for image 2
        std::pair<std::vector<cv::KeyPoint>, cv::Mat> result2 = processor_utils::detectAndCompute(config.detector, image2);
        std::vector<cv::KeyPoint> keypoints2 = result2.first;
        cv::Mat descriptors2 = result2.second;

        // Compute DSP descriptors for image 2 using factory system
        cv::Mat dspDescriptors2 = dspPoolingStrategy->computeDescriptors(image2, keypoints2, config.detector, config);

        // Match normal descriptors
        std::vector<cv::DMatch> matches = processor_utils::matchDescriptors(descriptors1, descriptors2, config.matchingStrategy);
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
        std::vector<cv::DMatch> topMatches = std::vector<cv::DMatch>(matches.begin(), matches.begin() + std::min(100, (int)matches.size()));

        // Match DSP descriptors
        std::vector<cv::DMatch> dspMatches = processor_utils::matchDescriptors(dspDescriptors1, dspDescriptors2, config.matchingStrategy);
        std::sort(dspMatches.begin(), dspMatches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
            return a.distance < b.distance;
        });
        std::vector<cv::DMatch> topDspMatches = std::vector<cv::DMatch>(dspMatches.begin(), dspMatches.begin() + std::min(50, (int)dspMatches.size()));

        // Visual comparison
        cv::Mat img_matches, dsp_img_matches;
        cv::drawMatches(image1, keypoints1, image2, keypoints2, topMatches, img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::drawMatches(image1, keypoints1, image2, keypoints2, topDspMatches, dsp_img_matches, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Show both matches side by side
        cv::imshow("Normal Descriptors Matches", img_matches);
        cv::imshow("DSP Descriptors Matches", dsp_img_matches);
        cv::waitKey(0); // Wait for a key press

        // Destroy the windows to clean up and move to the next iteration
        cv::destroyWindow("Normal Descriptors Matches");
        cv::destroyWindow("DSP Descriptors Matches");
    }
}

void VisualVerification::verifyHomography(const std::string& folder, 
                                         const std::string& results_folder, 
                                         const experiment_config& config) {
    // Read in 1.ppm as the reference image
    cv::Mat referenceImage = cv::imread(folder + "/1.ppm", cv::IMREAD_COLOR);

    // Detect keypoints and compute descriptors for the reference image
    std::vector<cv::KeyPoint> referenceKeypoints;
    cv::Mat referenceDescriptors;
    config.detector->detectAndCompute(referenceImage, cv::noArray(), referenceKeypoints, referenceDescriptors);

    // Sort the keypoints based on their response (strength)
    std::sort(referenceKeypoints.begin(), referenceKeypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response > b.response;
    });

    // Select the top 1000 keypoints (note: original comment said 100 but code used 1000)
    std::vector<cv::KeyPoint> top100Keypoints(referenceKeypoints.begin(), 
                                              referenceKeypoints.begin() + std::min(1000, static_cast<int>(referenceKeypoints.size())));

    // Extract the point coordinates from the top keypoints
    std::vector<cv::Point2f> top100Points;
    for (const auto& keypoint : top100Keypoints) {
        top100Points.push_back(keypoint.pt);
    }

    // Draw the keypoints on the reference image
    cv::Mat referenceImageWithKeypoints = referenceImage.clone();
    for (size_t i = 0; i < top100Keypoints.size(); i++) {
        const cv::KeyPoint& keypoint = top100Keypoints[i];
        cv::Scalar color = getKeypointColor(i);
        cv::drawKeypoints(referenceImageWithKeypoints, std::vector<cv::KeyPoint>{keypoint}, referenceImageWithKeypoints, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }

    // Iterate over the other images (2.ppm - 6.ppm)
    for (int i = 2; i <= 6; i++) {
        std::string imageFilename = folder + "/" + std::to_string(i) + ".ppm";
        cv::Mat image = cv::imread(imageFilename, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Failed to read image: " << imageFilename << std::endl;
            continue;
        }

        // Read the corresponding homography matrix
        std::string homographyFilename = folder + "/H_1_" + std::to_string(i);
        cv::Mat homography = processor_utils::readHomography(homographyFilename);

        // Project the top points from the reference image onto the current image
        std::vector<cv::Point2f> projectedPoints;
        cv::perspectiveTransform(top100Points, projectedPoints, homography);

        // Convert the projected points to cv::KeyPoint objects
        std::vector<cv::KeyPoint> projectedKeypoints;
        for (size_t j = 0; j < projectedPoints.size(); j++) {
            const cv::Point2f& point = projectedPoints[j];
            const cv::KeyPoint& referenceKeypoint = top100Keypoints[j];
            float size = referenceKeypoint.size;
            float angle = referenceKeypoint.angle;
            projectedKeypoints.emplace_back(cv::KeyPoint(point, size, angle));
        }

        // Draw the projected keypoints on the current image
        cv::Mat imageWithProjectedKeypoints = image.clone();
        for (size_t y = 0; y < projectedKeypoints.size(); y++) {
            const cv::KeyPoint& keypoint = projectedKeypoints[y];
            cv::Scalar color = getKeypointColor(y);
            cv::drawKeypoints(imageWithProjectedKeypoints, std::vector<cv::KeyPoint>{keypoint}, imageWithProjectedKeypoints, color, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }

        // Display the reference image with keypoints
        std::string referenceWindowName = "Reference Image - 1.ppm";
        cv::imshow(referenceWindowName, referenceImageWithKeypoints);

        // Display the current image with projected keypoints in a separate window
        std::string currentImageWindowName = "Projected Keypoints - " + std::to_string(i) + ".ppm";
        cv::imshow(currentImageWindowName, imageWithProjectedKeypoints);

        cv::waitKey(0); // Wait for a key press to continue

        // Close the displayed images windows
        cv::destroyWindow(referenceWindowName);
        cv::destroyWindow(currentImageWindowName);
    }
}