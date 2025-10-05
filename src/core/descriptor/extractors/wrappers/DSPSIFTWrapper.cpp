#include "DSPSIFTWrapper.hpp"
#include <algorithm>
#include <sstream>

namespace thesis_project::wrappers {

    DSPSIFTWrapper::DSPSIFTWrapper() {
        dspsift_ = DSPSIFT::create();
    }

    cv::Mat DSPSIFTWrapper::extract(const cv::Mat& image,
                                    const std::vector<cv::KeyPoint>& keypoints,
                                    const DescriptorParams& params) {
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> kps = keypoints;

        // Heuristic mapping: if scales provided, use them to derive DSP range
        int numScales = 1;
        double linePoint1 = 1.0, linePoint2 = 1.0;
        if (!params.scales.empty()) {
            numScales = static_cast<int>(params.scales.size());
            auto [minIt, maxIt] = std::minmax_element(params.scales.begin(), params.scales.end());
            linePoint1 = static_cast<double>(*minIt);
            linePoint2 = static_cast<double>(*maxIt);
        } else {
            // Sensible defaults (small pooling around 1.0)
            numScales = 3;
            linePoint1 = 0.85;
            linePoint2 = 1.30;
        }

        // Call DSPSIFT compute
        dspsift_->compute(image, kps, descriptors, numScales, linePoint1, linePoint2);
        return descriptors;
    }

    std::string DSPSIFTWrapper::getConfiguration() {
        std::stringstream ss;
        ss << "DSPSIFT Wrapper Configuration\n";
        return ss.str();
    }

}
