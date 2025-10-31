//-------------------------------------------------------------------------
// Name: CompositeDescriptorExtractor.hpp
// Description: Composite descriptor that combines multiple descriptor types
//              Supports averaging, max, min, and concatenation aggregation
//-------------------------------------------------------------------------

#ifndef COMPOSITE_DESCRIPTOR_EXTRACTOR_HPP
#define COMPOSITE_DESCRIPTOR_EXTRACTOR_HPP

#include "src/interfaces/IDescriptorExtractor.hpp"
#include "include/thesis_project/types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace thesis_project {

    /**
     * @brief Composite descriptor that combines multiple descriptors
     *
     * Extracts multiple descriptor types from the same keypoints and aggregates
     * them using configurable strategies (average, max, min, concatenate).
     *
     * Use Cases:
     * - Average SIFT + HardNet for robustness
     * - Max pooling across multiple CNN descriptors
     * - Concatenate descriptors for higher dimensionality
     *
     * Example YAML:
     * ```yaml
     * descriptors:
     *   - name: "sift_hardnet_avg"
     *     type: "composite"
     *     components:
     *       - descriptor: "sift"
     *         weight: 0.5
     *       - descriptor: "libtorch_hardnet"
     *         device: "cpu"
     *         weight: 0.5
     *     aggregation: "weighted_avg"
     * ```
     */
    class CompositeDescriptorExtractor : public IDescriptorExtractor {
    public:
        /**
         * @brief Configuration for a component descriptor
         */
        struct ComponentConfig {
            DescriptorType type;              ///< Descriptor type
            double weight = 1.0;              ///< Weight for weighted averaging (default 1.0)
            DescriptorParams params;          ///< Component-specific parameters

            ComponentConfig() = default;
            ComponentConfig(DescriptorType t, double w = 1.0)
                : type(t), weight(w) {}
        };

        /**
         * @brief Aggregation methods for combining descriptors
         */
        enum class AggregationMethod {
            AVERAGE,        ///< Simple average: (d1 + d2 + ... + dn) / n
            WEIGHTED_AVG,   ///< Weighted average: w1*d1 + w2*d2 + ... + wn*dn
            MAX,            ///< Element-wise maximum
            MIN,            ///< Element-wise minimum
            CONCATENATE     ///< Horizontal concatenation (increases dimension)
        };

        /**
         * @brief Construct composite descriptor extractor
         * @param components Vector of component descriptor configurations
         * @param aggregation Aggregation method to use
         */
        CompositeDescriptorExtractor(
            std::vector<ComponentConfig> components,
            AggregationMethod aggregation
        );

        /**
         * @brief Extract and aggregate descriptors from multiple components
         * @param image Input image
         * @param keypoints Keypoints to compute descriptors for
         * @param params Descriptor parameters (may be overridden by component params)
         * @return Aggregated descriptor matrix
         */
        cv::Mat extract(const cv::Mat& image,
                       const std::vector<cv::KeyPoint>& keypoints,
                       const DescriptorParams& params) override;

        /**
         * @brief Get descriptor size
         * For CONCATENATE: sum of component sizes
         * For others: size of first component (all must match)
         */
        int descriptorSize() const override;

        /**
         * @brief Get OpenCV descriptor type (CV_32F)
         */
        int descriptorType() const override { return CV_32F; }

        /**
         * @brief Get descriptor name
         */
        std::string name() const override;

        /**
         * @brief Convert aggregation method enum to string
         */
        static std::string aggregationMethodToString(AggregationMethod method);

        /**
         * @brief Convert string to aggregation method enum
         */
        static AggregationMethod stringToAggregationMethod(const std::string& str);

    private:
        /**
         * @brief Validate all descriptors have compatible dimensions
         * @param descriptors Vector of descriptor matrices
         * @throws std::runtime_error if dimensions incompatible for aggregation method
         */
        void validateDimensions(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Aggregate descriptors using configured method
         * @param descriptors Vector of descriptor matrices to aggregate
         * @return Aggregated descriptor matrix
         */
        cv::Mat aggregate(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform simple average aggregation
         */
        cv::Mat aggregateAverage(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform weighted average aggregation
         */
        cv::Mat aggregateWeightedAverage(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform element-wise maximum aggregation
         */
        cv::Mat aggregateMax(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform element-wise minimum aggregation
         */
        cv::Mat aggregateMin(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform horizontal concatenation
         */
        cv::Mat aggregateConcatenate(const std::vector<cv::Mat>& descriptors) const;

        std::vector<ComponentConfig> components_;                    ///< Component configurations
        std::vector<std::unique_ptr<IDescriptorExtractor>> extractors_; ///< Component extractors
        AggregationMethod aggregation_method_;                       ///< Aggregation method
        mutable int cached_descriptor_size_ = -1;                    ///< Cached descriptor size
    };

} // namespace thesis_project

#endif // COMPOSITE_DESCRIPTOR_EXTRACTOR_HPP
