//-------------------------------------------------------------------------
// Name: RGBSIFTChannelAverageWrapper.cpp
// Description: Implementation of RGBSIFT channel averaging wrapper
//-------------------------------------------------------------------------

#include "RGBSIFTChannelAverageWrapper.hpp"
#include "thesis_project/logging.hpp"
#include <stdexcept>

namespace thesis_project::wrappers {

    RGBSIFTChannelAverageWrapper::RGBSIFTChannelAverageWrapper() {
        // Create RGBSIFT extractor with default parameters
        rgbsift_ = cv::RGBSIFT::create();

        if (!rgbsift_) {
            throw std::runtime_error("Failed to create RGBSIFT extractor");
        }
    }

    cv::Mat RGBSIFTChannelAverageWrapper::extract(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        const DescriptorParams& params)
    {
        // Validate input
        if (image.empty()) {
            throw std::invalid_argument("RGBSIFTChannelAverageWrapper: Empty image");
        }

        if (keypoints.empty()) {
            LOG_WARNING("RGBSIFTChannelAverageWrapper: No keypoints provided, returning empty descriptor");
            return cv::Mat();
        }

        // Ensure color image for RGBSIFT
        cv::Mat color_image = image;
        if (image.channels() == 1) {
            LOG_WARNING("RGBSIFTChannelAverageWrapper: Grayscale image provided, converting to color");
            cv::cvtColor(image, color_image, cv::COLOR_GRAY2BGR);
        }

        // Extract RGBSIFT descriptors (384D)
        cv::Mat rgb_descriptors;
        rgbsift_->compute(color_image, const_cast<std::vector<cv::KeyPoint>&>(keypoints), rgb_descriptors);

        // Validate RGBSIFT output
        if (rgb_descriptors.empty()) {
            LOG_WARNING("RGBSIFTChannelAverageWrapper: RGBSIFT extraction failed, returning empty");
            return cv::Mat();
        }

        if (rgb_descriptors.cols != 384) {
            throw std::runtime_error(
                "RGBSIFTChannelAverageWrapper: Expected 384D descriptors, got " +
                std::to_string(rgb_descriptors.cols) + "D"
            );
        }

        // Split into RGB channels
        cv::Mat r_channel, g_channel, b_channel;
        splitChannels(rgb_descriptors, r_channel, g_channel, b_channel);

        // Average the channels
        cv::Mat averaged = averageChannels(r_channel, g_channel, b_channel);

        return averaged;
    }

    void RGBSIFTChannelAverageWrapper::splitChannels(
        const cv::Mat& rgb_desc,
        cv::Mat& r_channel,
        cv::Mat& g_channel,
        cv::Mat& b_channel) const
    {
        // RGBSIFT stores descriptors as [R0..R127 | G0..G127 | B0..B127] per keypoint
        // Extract each 128D channel using column ranges
        r_channel = rgb_desc(cv::Range::all(), cv::Range(0, 128)).clone();
        g_channel = rgb_desc(cv::Range::all(), cv::Range(128, 256)).clone();
        b_channel = rgb_desc(cv::Range::all(), cv::Range(256, 384)).clone();
    }

    cv::Mat RGBSIFTChannelAverageWrapper::averageChannels(
        const cv::Mat& r_channel,
        const cv::Mat& g_channel,
        const cv::Mat& b_channel) const
    {
        // Validate dimensions
        if (r_channel.size() != g_channel.size() || r_channel.size() != b_channel.size()) {
            throw std::runtime_error("RGBSIFTChannelAverageWrapper: Channel size mismatch");
        }

        if (r_channel.cols != 128) {
            throw std::runtime_error(
                "RGBSIFTChannelAverageWrapper: Expected 128D channels, got " +
                std::to_string(r_channel.cols) + "D"
            );
        }

        // Element-wise average: (R + G + B) / 3
        cv::Mat averaged;
        cv::addWeighted(r_channel, 1.0/3.0, g_channel, 1.0/3.0, 0.0, averaged);
        averaged += b_channel / 3.0;

        // Ensure output is CV_32F
        if (averaged.type() != CV_32F) {
            averaged.convertTo(averaged, CV_32F);
        }

        return averaged;
    }

} // namespace thesis_project::wrappers
