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
    namespace database {
        class DatabaseManager;  // Forward declaration
    }

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
            std::string keypoint_set_name;    ///< Optional: keypoint set override for this component (for paired sets)

            ComponentConfig() = default;
            ComponentConfig(const DescriptorType t, const double w = 1.0)
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
            CONCATENATE,    ///< Horizontal concatenation (increases dimension)
            CHANNEL_WISE    ///< Channel-wise fusion for grayscale + RGB (configurable output: 128D or 384D)
        };

        /**
         * @brief Output dimension mode for channel-wise fusion
         */
        enum class OutputDimensionMode {
            PRESERVE_RGB,   ///< Output 384D (R128 + G128 + B128) - full color information
            COLLAPSE_GRAY   ///< Output 128D (average RGB channels back to grayscale)
        };

        /**
         * @brief Construct composite descriptor extractor
         * @param components Vector of component descriptor configurations
         * @param aggregation Aggregation method to use
         * @param output_mode Output dimension mode (only used for CHANNEL_WISE aggregation)
         */
        CompositeDescriptorExtractor(
            std::vector<ComponentConfig> components,
            AggregationMethod aggregation,
            OutputDimensionMode output_mode = OutputDimensionMode::COLLAPSE_GRAY
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
         * @brief Set database context for component-specific keypoint loading
         *
         * This must be called before extract() when using paired keypoint sets.
         *
         * @param db_manager Pointer to database manager (or nullptr to disable)
         * @param scene_name Current scene name
         * @param image_name Current image name
         */
        static void setDatabaseContext(
            thesis_project::database::DatabaseManager* db_manager,
            const std::string& scene_name,
            const std::string& image_name
        );

        /**
         * @brief Check if components use different keypoint sets (paired mode)
         * @return true if components have different keypoint_set_name values
         */
        bool usesPairedKeypointSets() const;

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
        static cv::Mat aggregateAverage(const std::vector<cv::Mat>& descriptors);

        /**
         * @brief Perform weighted average aggregation
         */
        cv::Mat aggregateWeightedAverage(const std::vector<cv::Mat>& descriptors) const;

        /**
         * @brief Perform element-wise maximum aggregation
         */
        static cv::Mat aggregateMax(const std::vector<cv::Mat>& descriptors);

        /**
         * @brief Perform element-wise minimum aggregation
         */
        static cv::Mat aggregateMin(const std::vector<cv::Mat>& descriptors);

        /**
         * @brief Perform horizontal concatenation
         */
        static cv::Mat aggregateConcatenate(const std::vector<cv::Mat>& descriptors);

        /**
         * @brief Perform channel-wise fusion for grayscale + RGB descriptors
         *
         * For SIFT (128D) + RGBSIFT (384D):
         * 1. Broadcast SIFT to all 3 channels
         * 2. Fuse with each RGBSIFT channel (R, G, B) separately
         * 3. Output either 384D (PRESERVE_RGB) or 128D (COLLAPSE_GRAY)
         *
         * @param descriptors Vector containing [grayscale_desc, rgb_desc]
         * @return Fused descriptor (128D or 384D based on output_mode_)
         */
        cv::Mat aggregateChannelWise(const std::vector<cv::Mat>& descriptors) const;

        std::vector<ComponentConfig> components_;                    ///< Component configurations
        std::vector<std::unique_ptr<IDescriptorExtractor>> extractors_; ///< Component extractors
        AggregationMethod aggregation_method_;                       ///< Aggregation method
        OutputDimensionMode output_mode_;                            ///< Output dimension mode (for CHANNEL_WISE)
        mutable int cached_descriptor_size_ = -1;                    ///< Cached descriptor size

        struct ThreadLocalContext {
            thesis_project::database::DatabaseManager* db_manager = nullptr;
            std::string scene_name;
            std::string image_name;
        };

        static ThreadLocalContext& threadContext();
    };

} // namespace thesis_project

#endif // COMPOSITE_DESCRIPTOR_EXTRACTOR_HPP
