#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "keypoints/HoNC.h"
#include <memory>

namespace thesis_project::wrappers {

    class HoNCWrapper : public IDescriptorExtractor {
    private:
        std::unique_ptr<HoNC> honc_;

    public:
        HoNCWrapper();

        cv::Mat extract(const cv::Mat& image,
                        const std::vector<cv::KeyPoint>& keypoints,
                        const DescriptorParams& params = {}) override;

        std::string name() const override { return "HoNC"; }
        int descriptorSize() const override { return 128; }
        int descriptorType() const override { return static_cast<int>(thesis_project::DescriptorType::HoNC); }

        std::string getConfiguration() const;
    };

}
