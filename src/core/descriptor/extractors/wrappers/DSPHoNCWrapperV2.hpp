//-------------------------------------------------------------------------
// Name: DSPHoNCWrapperV2.hpp
// Description: IDescriptorExtractor wrapper for HoNC with pyramid-aware DSP
//              Uses native HoNC DSP operator() for proper color pyramid handling
//-------------------------------------------------------------------------

#ifndef DSPHONC_WRAPPER_V2_HPP
#define DSPHONC_WRAPPER_V2_HPP

#include "keypoints/HoNC.h"
#include "keypoints/DSPVanillaSIFTWrapper.h"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"

namespace thesis_project::wrappers {

    /**
 * @brief Pyramid-aware DSP wrapper for HoNC
 *
 * HoNC (Histogram of Normalized Colors) uses:
 * - createInitialColorImage() for BGR pyramid base
 * - Color Gaussian pyramid (BGR, not HSV)
 * - 2x2x2 RGB histogram = 8 bins (128 dimensions)
 * - Color normalization with bias/gain adjustment
 */
    class DSPHoNCWrapperV2 : public IDescriptorExtractor {
    public:
        DSPHoNCWrapperV2();

        cv::Mat extract(const cv::Mat& image,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params) override;

        int descriptorSize() const override { return last_descriptor_size_; }
        int descriptorType() const override { return CV_32F; }
        std::string name() const override { return "DSPHONC_V2"; }

    private:
        cv::Ptr<DSPVanillaSIFTWrapper<HoNC>> wrapper_;
        mutable int last_descriptor_size_ = 128;
    };

}

#endif // DSPHONC_WRAPPER_V2_HPP
