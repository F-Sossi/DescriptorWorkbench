#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "keypoints/VanillaSIFT.h"
#include <memory>

namespace thesis_project::wrappers {

    class VSIFTWrapper : public IDescriptorExtractor {
    private:
        cv::Ptr<VanillaSIFT> vsift_;

    public:
        VSIFTWrapper();

        cv::Mat extract(const cv::Mat& image,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params = {}) override;

        std::string name() const override { return "vSIFT"; }
        int descriptorSize() const override { return 128; }
        int descriptorType() const override { return static_cast<int>(thesis_project::DescriptorType::vSIFT); }

        std::string getConfiguration() const;
    };

}
