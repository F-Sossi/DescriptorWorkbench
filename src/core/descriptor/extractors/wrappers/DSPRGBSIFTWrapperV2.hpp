//-------------------------------------------------------------------------
// Name: DSPRGBSIFTWrapperV2.hpp
// Description: IDescriptorExtractor wrapper for RGBSIFT with pyramid-aware DSP
//              Uses DSPVanillaSIFTWrapper template with RGBSIFT specialization
//-------------------------------------------------------------------------

#ifndef DSPRGBSIFT_WRAPPER_V2_HPP
#define DSPRGBSIFT_WRAPPER_V2_HPP

#include "keypoints/RGBSIFT.h"
#include "keypoints/DSPVanillaSIFTWrapper.h"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"

namespace thesis_project::wrappers {

    /**
 * @brief Pyramid-aware DSP wrapper for RGBSIFT
 *
 * This wrapper uses template specialization with DSPVanillaSIFTWrapper<RGBSIFT>
 * BUT we need to handle the color pyramid specially. RGBSIFT uses:
 * - createInitialColorImage() for RGB pyramid base
 * - Color Gaussian pyramid (not grayscale)
 * - Stacked [R|G|B] descriptors (384 dimensions)
 */
    class DSPRGBSIFTWrapperV2 : public IDescriptorExtractor {
    public:
        DSPRGBSIFTWrapperV2();

        cv::Mat extract(const cv::Mat& image,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params) override;

        int descriptorSize() const override { return last_descriptor_size_; }
        int descriptorType() const override { return CV_32F; }
        std::string name() const override { return "DSPRGBSIFT_V2"; }

    private:
        cv::Ptr<DSPVanillaSIFTWrapper<cv::RGBSIFT>> wrapper_;
        mutable int last_descriptor_size_ = 384;
    };

}

#endif // DSPRGBSIFT_WRAPPER_V2_HPP
