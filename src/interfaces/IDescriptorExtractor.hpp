#ifndef THESIS_PROJECT_IDESCRIPTOR_EXTRACTOR_HPP
#define THESIS_PROJECT_IDESCRIPTOR_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include "thesis_project/types.hpp"

namespace thesis_project {

    // Use DescriptorParams from types.hpp

    /**
     * @brief Interface for descriptor extraction algorithms
     */
    class IDescriptorExtractor {
    public:
        virtual ~IDescriptorExtractor() = default;

        /**
         * @brief Extract descriptors from keypoints
         */
        virtual cv::Mat extract(const cv::Mat& image,
                               const std::vector<cv::KeyPoint>& keypoints,
                               const DescriptorParams& params) = 0;

        /**
         * @brief Get the descriptor name
         */
        [[nodiscard]] virtual std::string name() const = 0;

        /**
         * @brief Get the descriptor size (dimensions)
         */
        [[nodiscard]] virtual int descriptorSize() const = 0;

        /**
         * @brief Get the descriptor type
         */
        [[nodiscard]] virtual int descriptorType() const = 0;
    };

} // namespace thesis_project

#endif // THESIS_PROJECT_IDESCRIPTOR_EXTRACTOR_HPP
