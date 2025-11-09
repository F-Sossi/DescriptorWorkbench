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
        AggregationMethod aggregation,
        OutputDimensionMode output_mode)
        : components_(std::move(components))
        , aggregation_method_(aggregation)
        , output_mode_(output_mode)
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
            return {};
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

        // For non-concatenate and non-channel-wise aggregation, all must have same dimension
        if (aggregation_method_ != AggregationMethod::CONCATENATE &&
            aggregation_method_ != AggregationMethod::CHANNEL_WISE) {
            const int descriptor_dim = descriptors[0].cols;
            for (size_t i = 1; i < descriptors.size(); ++i) {
                if (descriptors[i].cols != descriptor_dim) {
                    // Build descriptive error with component names and dimensions
                    std::ostringstream error_msg;
                    error_msg << "CompositeDescriptorExtractor: Descriptor dimension mismatch for '"
                              << aggregationMethodToString(aggregation_method_) << "' aggregation.\n";
                    error_msg << "  Component 0 (" << extractors_[0]->name() << "): "
                              << descriptors[0].cols << "D\n";
                    error_msg << "  Component " << i << " (" << extractors_[i]->name() << "): "
                              << descriptors[i].cols << "D\n";
                    error_msg << "  All components:\n";
                    for (size_t j = 0; j < extractors_.size(); ++j) {
                        error_msg << "    [" << j << "] " << extractors_[j]->name()
                                  << " (" << descriptors[j].cols << "D)\n";
                    }
                    error_msg << "  Solution: Use 'concatenate' aggregation for different dimensions, "
                              << "or ensure all descriptors return the same dimension.";
                    throw std::runtime_error(error_msg.str());
                }
            }
        }

        // For channel-wise, validate we have exactly 2 components with expected dimensions
        if (aggregation_method_ == AggregationMethod::CHANNEL_WISE) {
            if (descriptors.size() != 2) {
                throw std::runtime_error(
                    "CompositeDescriptorExtractor: CHANNEL_WISE aggregation requires exactly 2 components, got " +
                    std::to_string(descriptors.size())
                );
            }
            // Expect 128D (grayscale) and 384D (RGB) descriptors
            const bool has_128d = (descriptors[0].cols == 128 || descriptors[1].cols == 128);
            if (const bool has_384d = (descriptors[0].cols == 384 || descriptors[1].cols == 384); !has_128d || !has_384d) {
                throw std::runtime_error(
                    "CompositeDescriptorExtractor: CHANNEL_WISE aggregation expects 128D and 384D descriptors, got " +
                    std::to_string(descriptors[0].cols) + "D and " + std::to_string(descriptors[1].cols) + "D"
                );
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
            case AggregationMethod::CHANNEL_WISE:
                return aggregateChannelWise(descriptors);
            default:
                throw std::runtime_error("Unknown aggregation method");
        }
    }

    cv::Mat CompositeDescriptorExtractor::aggregateAverage(
        const std::vector<cv::Mat>& descriptors) {
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
        const std::vector<cv::Mat>& descriptors) {
        // Element-wise maximum
        cv::Mat result = descriptors[0].clone();

        for (size_t i = 1; i < descriptors.size(); ++i) {
            cv::max(result, descriptors[i], result);
        }

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateMin(
        const std::vector<cv::Mat>& descriptors) {
        // Element-wise minimum
        cv::Mat result = descriptors[0].clone();

        for (size_t i = 1; i < descriptors.size(); ++i) {
            cv::min(result, descriptors[i], result);
        }

        return result;
    }

    cv::Mat CompositeDescriptorExtractor::aggregateConcatenate(
        const std::vector<cv::Mat>& descriptors) {
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
        } else if (aggregation_method_ == AggregationMethod::CHANNEL_WISE) {
            // Channel-wise fusion output depends on output mode
            if (output_mode_ == OutputDimensionMode::PRESERVE_RGB) {
                cached_descriptor_size_ = 384;  // R128 + G128 + B128
            } else {
                cached_descriptor_size_ = 128;  // Averaged back to grayscale
            }
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
            case AggregationMethod::CHANNEL_WISE: return "channel_wise";
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
        if (str == "channel_wise" || str == "channelwise" || str == "channel") return AggregationMethod::CHANNEL_WISE;

        throw std::runtime_error("Unknown aggregation method: " + str);
    }

    cv::Mat CompositeDescriptorExtractor::aggregateChannelWise(
        const std::vector<cv::Mat>& descriptors) const
    {
        // Channel-wise fusion for grayscale (128D) + RGB (384D) descriptors
        // Validated in validateDimensions: exactly 2 components, one 128D and one 384D

        // Determine which is grayscale and which is RGB
        cv::Mat gray_desc, rgb_desc;
        if (descriptors[0].cols == 128) {
            gray_desc = descriptors[0];
            rgb_desc = descriptors[1];
        } else {
            gray_desc = descriptors[1];
            rgb_desc = descriptors[0];
        }

        int num_keypoints = gray_desc.rows;

        // Split RGB descriptor into R, G, B channels (each 128D)
        cv::Mat r_channel = rgb_desc(cv::Range::all(), cv::Range(0, 128));
        cv::Mat g_channel = rgb_desc(cv::Range::all(), cv::Range(128, 256));
        cv::Mat b_channel = rgb_desc(cv::Range::all(), cv::Range(256, 384));

        // Fuse grayscale descriptor with each RGB channel
        // Using weighted averaging with equal weights (can be extended later)
        cv::Mat r_fused, g_fused, b_fused;
        cv::addWeighted(gray_desc, 0.5, r_channel, 0.5, 0.0, r_fused);
        cv::addWeighted(gray_desc, 0.5, g_channel, 0.5, 0.0, g_fused);
        cv::addWeighted(gray_desc, 0.5, b_channel, 0.5, 0.0, b_fused);

        // Output based on configured mode
        if (output_mode_ == OutputDimensionMode::PRESERVE_RGB) {
            // Output 384D: [R_fused | G_fused | B_fused]
            cv::Mat result;
            cv::hconcat(std::vector<cv::Mat>{r_fused, g_fused, b_fused}, result);
            return result;
        } else {
            // Output 128D: average the fused channels back to grayscale
            cv::Mat result = (r_fused + g_fused + b_fused) / 3.0;
            return result;
        }
    }

} // namespace thesis_project
