//-------------------------------------------------------------------------
// Name: DSPHoWHWrapperV2.hpp
// Description: IDescriptorExtractor wrapper for HoWH with pyramid-aware DSP
//              Uses native HoWH DSP operator() for proper color pyramid handling
//-------------------------------------------------------------------------

#ifndef DSPHOWH_WRAPPER_V2_HPP
#define DSPHOWH_WRAPPER_V2_HPP

#include "keypoints/HoWH.h"
#include "keypoints/DSPVanillaSIFTWrapper.h"
#include "src/interfaces/IDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"

namespace thesis_project {
namespace wrappers {

/**
 * @brief Pyramid-aware DSP wrapper for HoWH
 *
 * HoWH (Histogram of Weighted Hue) uses:
 * - createInitialColorImage() for BGR pyramid base
 * - Color Gaussian pyramid converted to HSV
 * - Hue weighted by saturation (128 dimensions)
 * - 0.6 weight factor for descriptor stacking compatibility
 */
class DSPHoWHWrapperV2 : public IDescriptorExtractor {
public:
    DSPHoWHWrapperV2();

    cv::Mat extract(const cv::Mat& image,
                    const std::vector<cv::KeyPoint>& keypoints,
                    const DescriptorParams& params) override;

    int descriptorSize() const override { return last_descriptor_size_; }
    int descriptorType() const override { return CV_32F; }
    std::string name() const override { return "DSPHOWH_V2"; }

private:
    cv::Ptr<DSPVanillaSIFTWrapper<HoWH>> wrapper_;
    mutable int last_descriptor_size_ = 128;
};

} // namespace wrappers
} // namespace thesis_project

#endif // DSPHOWH_WRAPPER_V2_HPP
