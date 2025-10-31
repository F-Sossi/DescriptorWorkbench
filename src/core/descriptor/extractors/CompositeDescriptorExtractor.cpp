//-------------------------------------------------------------------------
// Name: CompositeDescriptorExtractor.cpp
// Description: Implementation of composite descriptor extractor
//-------------------------------------------------------------------------

#include "CompositeDescriptorExtractor.hpp"
#include "src/core/descriptor/factories/DescriptorFactory.hpp"
#include "thesis_project/logging.hpp"
#include <stdexcept>
#include <sstream>
#include <numeric>

namespace thesis_project {

    CompositeDescriptorExtractor::CompositeDescriptorExtractor(
        std::vector<ComponentConfig> components,
        AggregationMethod aggregation)
        : components_(std::move(components))
        , aggregation_method_(aggregation)
    {
        // Validate we have at least 2 components
        if (components_.size() < 2) {
            throw std::invalid_argument(
                "CompositeDescriptorExtractor requires at least 2 components, got " +
                std::to_string(components_.size())
            );
        }

        // Validate weights for weighted averaging
        if (aggregation_method_ == AggregationMethod::WEIGHTED_AVG) {
            double weight_sum = 0.0;
            for (const auto& comp : components_) {
                weight_sum += comp.weight;
            }

            // Normalize weights to sum to 1.0
            if (std::abs(weight_sum - 1.0) > 1e-6) {
                LOG_WARNING(
                    "Composite descriptor weights sum to " + std::to_string(weight_sum) +
                    ", normalizing to 1.0"
                );
                for (auto& comp : components_) {
                    comp.weight /= weight_sum;
                }
            }
        }

        // Create extractors for each component
        extractors_.reserve(components_.size());
        for (const auto& comp : components_) {
            try {
                auto extractor = factories::DescriptorFactory::create(comp.type);
                if (!extractor) {
                    throw std::runtime_error(
                        "Failed to create extractor for descriptor type " +
                        std::to_string(static_cast<int>(comp.type))
                    );
                }
                extractors_.push_back(std::move(extractor));
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "CompositeDescriptorExtractor: Failed to create component extractor: " +
                    std::string(e.what())
                );
            }
        }

        LOG_INFO(
            "CompositeDescriptorExtractor created with " +
            std::to_string(components_.size()) + " components, aggregation: " +
            aggregationMethodToString(aggregation_method_)
        );
    }

    cv::Mat CompositeDescriptorExtractor::extract(
        const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints,
        const DescriptorParams& params)
    {
        // Validate inputs
        if (image.empty()) {
            throw std::invalid_argument("CompositeDescriptorExtractor: Empty image");
        }

        if (keypoints.empty()) {
            LOG_WARNING("CompositeDescriptorExtractor: No keypoints provided");
            return cv::Mat();
        }

        // Extract descriptors from all components
        std::vector<cv::Mat> descriptors;
        descriptors.reserve(extractors_.size());

        for (size_t i = 0; i < extractors_.size(); ++i) {
            try {
                // Use component-specific params if provided, otherwise use global params
                const DescriptorParams& comp_params =
                    components_[i].params.pooling != PoolingStrategy::NONE
                    ? components_[i].params
                    : params;

                cv::Mat desc = extractors_[i]->extract(image, keypoints, comp_params);

                if (desc.empty()) {
                    throw std::runtime_error(
                        "Component " + std::to_string(i) + " returned empty descriptors"
                    );
                }

                descriptors.push_back(desc);
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "CompositeDescriptorExtractor: Component " + std::to_string(i) +
                    " extraction failed: " + std::string(e.what())
                );
            }
        }

        // Validate dimensions
        validateDimensions(descriptors);

        // Aggregate descriptors
        cv::Mat aggregated = aggregate(descriptors);

        return aggregated;
    }

    void CompositeDescriptorExtractor::validateDimensions(
        const std::vector<cv::Mat>& descriptors) const
    {
        if (descriptors.empty()) {
            throw std::runtime_error("CompositeDescriptorExtractor: No descriptors to validate");
        }

        // All descriptors must have same number of rows (keypoints)
        int num_keypoints = descriptors[0].rows;
        for (size_t i = 1; i < descriptors.size(); ++i) {
            if (descriptors[i].rows != num_keypoints) {
                throw std::runtime_error(
                    std::string("CompositeDescriptorExtractor: Keypoint count mismatch. ") +
                    "Component 0: " + std::to_string(descriptors[0].rows) +
                    ", Component " + std::to_string(i) + ": " + std::to_string(descriptors[i].rows)
                );
            }
        }

        // For non-concatenate aggregation, all must have same dimension
        if (aggregation_method_ != AggregationMethod::CONCATENATE) {
            int descriptor_dim = descriptors[0].cols;
            for (size_t i = 1; i < descriptors.size(); ++i) {
                if (descriptors[i].cols != descriptor_dim) {
                    throw std::runtime_error(
                        std::string("CompositeDescriptorExtractor: Descriptor dimension mismatch for ") +
                        aggregationMethodToString(aggregation_method_) + " aggregation. " +
                        "Component 0: " + std::to_string(descriptors[0].cols) + "D, " +
                        "Component " + std::to_string(i) + ": " + std::to_string(descriptors[i].cols) + "D. " +
                        "Use 'concatenate' aggregation for different dimensions or ensure all descriptors are same size."
                    );
                }
            }
        }

        // All must be CV_32F
        for (size_t i = 0; i < descriptors.size(); ++i) {
            if (descriptors[i].type() != CV_32F) {
                throw std::runtime_error(
                    "CompositeDescriptorExtractor: Component " + std::to_string(i) +
                    " has invalid type (expected CV_32F)"
                );
            }
        }
    }

    cv::Mat CompositeDescriptorExtractor::aggregate(
        const std::vector<cv::Mat>& descriptors) const
    {
        switch (aggregation_method_) {
            case AggregationMethod::AVERAGE:
                return aggregateAverage(descriptors);
            case AggregationMethod::WEIGHTED_AVG:
                return aggregateWeightedAverage(descriptors);
            case AggregationMethod::MAX:
                return aggregateMax(descriptors);
            case AggregationMethod::MIN:
                return aggregateMin(descriptors);
            case AggregationMethod::CONCATENATE:
                return aggregateConcatenate(descriptors);
            default:
                throw std::runtime_error("Unknown aggregation method");
        }
    }

    cv::Mat CompositeDescriptorExtractor::aggregateAverage(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Simple average: sum all descriptors and divide by count
        cv::Mat result = descriptors[0].clone();

        for (size_t i = 1; i < descriptors.size(); ++i) {
            result += descriptors[i];
        }

        result /= static_cast<double>(descriptors.size());

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateWeightedAverage(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Weighted average: w1*d1 + w2*d2 + ... + wn*dn
        cv::Mat result = cv::Mat::zeros(descriptors[0].size(), CV_32F);

        for (size_t i = 0; i < descriptors.size(); ++i) {
            result += components_[i].weight * descriptors[i];
        }

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateMax(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Element-wise maximum
        cv::Mat result = descriptors[0].clone();

        for (size_t i = 1; i < descriptors.size(); ++i) {
            cv::max(result, descriptors[i], result);
        }

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateMin(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Element-wise minimum
        cv::Mat result = descriptors[0].clone();

        for (size_t i = 1; i < descriptors.size(); ++i) {
            cv::min(result, descriptors[i], result);
        }

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateConcatenate(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Horizontal concatenation
        cv::Mat result;
        cv::hconcat(descriptors, result);
        return result;
    }

    int CompositeDescriptorExtractor::descriptorSize() const
    {
        if (cached_descriptor_size_ != -1) {
            return cached_descriptor_size_;
        }

        if (aggregation_method_ == AggregationMethod::CONCATENATE) {
            // Sum of all component sizes
            int total_size = 0;
            for (const auto& extractor : extractors_) {
                total_size += extractor->descriptorSize();
            }
            cached_descriptor_size_ = total_size;
        } else {
            // All components must have same size (validated during extraction)
            // Use first component's size
            cached_descriptor_size_ = extractors_[0]->descriptorSize();
        }

        return cached_descriptor_size_;
    }

    std::string CompositeDescriptorExtractor::name() const
    {
        std::ostringstream oss;
        oss << "Composite[";

        for (size_t i = 0; i < extractors_.size(); ++i) {
            if (i > 0) oss << "+";
            oss << extractors_[i]->name();
        }

        oss << "|" << aggregationMethodToString(aggregation_method_) << "]";

        return oss.str();
    }

    std::string CompositeDescriptorExtractor::aggregationMethodToString(AggregationMethod method)
    {
        switch (method) {
            case AggregationMethod::AVERAGE: return "average";
            case AggregationMethod::WEIGHTED_AVG: return "weighted_avg";
            case AggregationMethod::MAX: return "max";
            case AggregationMethod::MIN: return "min";
            case AggregationMethod::CONCATENATE: return "concatenate";
            default: return "unknown";
        }
    }

    CompositeDescriptorExtractor::AggregationMethod
    CompositeDescriptorExtractor::stringToAggregationMethod(const std::string& str)
    {
        if (str == "average" || str == "avg") return AggregationMethod::AVERAGE;
        if (str == "weighted_avg" || str == "weighted_average") return AggregationMethod::WEIGHTED_AVG;
        if (str == "max") return AggregationMethod::MAX;
        if (str == "min") return AggregationMethod::MIN;
        if (str == "concatenate" || str == "concat" || str == "stack") return AggregationMethod::CONCATENATE;

        throw std::runtime_error("Unknown aggregation method: " + str);
    }

} // namespace thesis_project
