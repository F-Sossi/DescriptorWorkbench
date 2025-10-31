//-------------------------------------------------------------------------
// Name: RGBSIFTChannelAverageWrapper.hpp
// Description: IDescriptorExtractor wrapper that averages RGBSIFT RGB channels
//              Converts 384D RGBSIFT (3×128D) to 128D by averaging R, G, B channels
//-------------------------------------------------------------------------

#ifndef RGBSIFT_CHANNEL_AVERAGE_WRAPPER_HPP
#define RGBSIFT_CHANNEL_AVERAGE_WRAPPER_HPP

#include "keypoints/RGBSIFT.h"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace thesis_project::wrappers {

    /**
     * @brief RGBSIFT wrapper that averages RGB channels to reduce dimensionality
     *
     * RGBSIFT produces 384D descriptors (3×128D for R, G, B channels).
     * This wrapper averages the three channels element-wise to produce
     * a 128D descriptor compatible with standard SIFT, HardNet, etc.
     *
     * Algorithm:
     * 1. Extract RGBSIFT descriptor → [N × 384]
     * 2. Split into R, G, B channels → 3 × [N × 128]
     * 3. Average: result = (R + G + B) / 3 → [N × 128]
     *
     * Use Cases:
     * - Standalone: Use RGBSIFT as 128D descriptor
     * - Composite: Combine with other 128D descriptors (SIFT, HardNet)
     */
    class RGBSIFTChannelAverageWrapper : public IDescriptorExtractor {
    public:
        RGBSIFTChannelAverageWrapper();

        /**
         * @brief Extract RGBSIFT and average RGB channels
         * @param image Input image (should be color)
         * @param keypoints Keypoints to compute descriptors for
         * @param params Descriptor parameters
         * @return Averaged 128D descriptors [N × 128]
         */
        cv::Mat extract(const cv::Mat& image,
                       const std::vector<cv::KeyPoint>& keypoints,
                       const DescriptorParams& params) override;

        /**
         * @brief Get descriptor size (always 128 after averaging)
         */
        int descriptorSize() const override { return 128; }

        /**
         * @brief Get OpenCV descriptor type (CV_32F)
         */
        int descriptorType() const override { return CV_32F; }

        /**
         * @brief Get descriptor name
         */
        std::string name() const override { return "RGBSIFT_CHANNEL_AVG"; }

    private:
        /**
         * @brief Split 384D RGBSIFT descriptor into R, G, B channels
         * @param rgb_desc Input RGBSIFT descriptors [N × 384]
         * @param r_channel Output R channel [N × 128]
         * @param g_channel Output G channel [N × 128]
         * @param b_channel Output B channel [N × 128]
         */
        void splitChannels(const cv::Mat& rgb_desc,
                          cv::Mat& r_channel,
                          cv::Mat& g_channel,
                          cv::Mat& b_channel) const;

        /**
         * @brief Average three channel matrices element-wise
         * @param r_channel R channel [N × 128]
         * @param g_channel G channel [N × 128]
         * @param b_channel B channel [N × 128]
         * @return Averaged descriptor [N × 128]
         */
        cv::Mat averageChannels(const cv::Mat& r_channel,
                               const cv::Mat& g_channel,
                               const cv::Mat& b_channel) const;

        cv::Ptr<cv::RGBSIFT> rgbsift_;  ///< Internal RGBSIFT extractor
    };

} // namespace thesis_project::wrappers

#endif // RGBSIFT_CHANNEL_AVERAGE_WRAPPER_HPP
