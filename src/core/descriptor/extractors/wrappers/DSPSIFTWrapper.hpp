#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "keypoints/DSPSIFT.h"
#include <memory>

namespace thesis_project::wrappers {

    class DSPSIFTWrapper : public IDescriptorExtractor {
    private:
        cv::Ptr<DSPSIFT> dspsift_;

    public:
        DSPSIFTWrapper();

        cv::Mat extract(const cv::Mat& image,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params = {}) override;

        std::string name() const override { return "DSPSIFT"; }
        int descriptorSize() const override { return 128; }
        int descriptorType() const override { return static_cast<int>(thesis_project::DescriptorType::DSPSIFT); }

        static std::string getConfiguration();
    };

}
