#pragma once

#include "thesis_project/types.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace thesis_project {

    // Interface for keypoint generation
    // This will be implemented in Stage 3
    class IKeypointGenerator {
    public:
        virtual ~IKeypointGenerator() = default;
        
        // TODO: Add interface methods in Stage 3
        // virtual std::vector<cv::KeyPoint> detect(const cv::Mat& image,
        //                                         const KeypointParams& params = {}) = 0;
        // virtual std::string name() const = 0;
    };

} // namespace thesis_project