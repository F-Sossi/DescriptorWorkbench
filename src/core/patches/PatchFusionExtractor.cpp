#include "PatchFusionExtractor.hpp"
#include <opencv2/core.hpp>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <sstream>

namespace thesis_project {
namespace patches {

PatchFusionExtractor::PatchFusionExtractor(
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components,
    PatchFusionMethod method,
    const std::vector<float>& weights,
    const std::string& name_override)
    : components_(std::move(components)),
      method_(method),
      weights_(weights) {

    if (components_.empty()) {
        throw std::invalid_argument("PatchFusionExtractor: at least one component required");
    }

    // Validate weights for WEIGHTED_AVG
    if (method_ == PatchFusionMethod::WEIGHTED_AVG) {
        if (weights_.empty()) {
            // Default to equal weights
            weights_.resize(components_.size(), 1.0f / static_cast<float>(components_.size()));
        } else if (weights_.size() != components_.size()) {
            throw std::invalid_argument("PatchFusionExtractor: weights count must match component count");
        }
    }

    // Validate dimensions for non-concatenate methods
    if (method_ != PatchFusionMethod::CONCATENATE) {
        int first_dim = components_[0]->descriptorSize();
        for (size_t i = 1; i < components_.size(); ++i) {
            if (components_[i]->descriptorSize() != first_dim) {
                // For CHANNEL_WISE, we allow 128 + 384 combinations
                if (method_ == PatchFusionMethod::CHANNEL_WISE) {
                    continue;  // Will handle specially
                }
                throw std::invalid_argument(
                    "PatchFusionExtractor: dimension mismatch for " +
                    fusionMethodToString(method_) + " fusion. Component 0: " +
                    std::to_string(first_dim) + ", Component " + std::to_string(i) +
                    ": " + std::to_string(components_[i]->descriptorSize()));
            }
        }
    }

    output_dim_ = computeOutputDimension();
    name_ = name_override.empty() ? generateName() : name_override;
}

cv::Mat PatchFusionExtractor::extractFromPatches(
    const std::vector<cv::Mat>& patches,
    const DescriptorParams& params) {

    if (patches.empty()) {
        return cv::Mat();
    }

    // Extract descriptors from each component
    std::vector<cv::Mat> component_descs;
    component_descs.reserve(components_.size());

    for (auto& component : components_) {
        cv::Mat desc = component->extractFromPatches(patches, params);
        component_descs.push_back(desc);
    }

    // Fuse the descriptors
    return fuseDescriptors(component_descs);
}

cv::Mat PatchFusionExtractor::fuseDescriptors(const std::vector<cv::Mat>& component_descs) const {
    if (component_descs.empty()) {
        return cv::Mat();
    }

    const int num_patches = component_descs[0].rows;

    // Verify all components have same number of patches
    for (const auto& desc : component_descs) {
        if (desc.rows != num_patches) {
            throw std::runtime_error("PatchFusionExtractor: component descriptor count mismatch");
        }
    }

    cv::Mat result;

    switch (method_) {
        case PatchFusionMethod::CONCATENATE: {
            // Horizontal concatenation
            cv::hconcat(component_descs, result);
            break;
        }

        case PatchFusionMethod::AVERAGE: {
            result = cv::Mat::zeros(num_patches, component_descs[0].cols, CV_32F);
            for (const auto& desc : component_descs) {
                result += desc;
            }
            result /= static_cast<float>(component_descs.size());
            break;
        }

        case PatchFusionMethod::WEIGHTED_AVG: {
            result = cv::Mat::zeros(num_patches, component_descs[0].cols, CV_32F);
            for (size_t i = 0; i < component_descs.size(); ++i) {
                result += weights_[i] * component_descs[i];
            }
            break;
        }

        case PatchFusionMethod::MAX: {
            result = component_descs[0].clone();
            for (size_t i = 1; i < component_descs.size(); ++i) {
                cv::max(result, component_descs[i], result);
            }
            break;
        }

        case PatchFusionMethod::MIN: {
            result = component_descs[0].clone();
            for (size_t i = 1; i < component_descs.size(); ++i) {
                cv::min(result, component_descs[i], result);
            }
            break;
        }

        case PatchFusionMethod::CHANNEL_WISE: {
            // Special handling for 128D + 384D -> 128D or 384D
            // This averages corresponding channels across descriptors

            // Find the minimum dimension among components
            int min_dim = component_descs[0].cols;
            for (const auto& desc : component_descs) {
                min_dim = std::min(min_dim, desc.cols);
            }

            // Use minimum dimension as output (typically 128D)
            result = cv::Mat::zeros(num_patches, min_dim, CV_32F);

            for (const auto& desc : component_descs) {
                // For 384D descriptors, average 3 channels into 1
                if (desc.cols == 3 * min_dim) {
                    for (int r = 0; r < num_patches; ++r) {
                        const float* src = desc.ptr<float>(r);
                        float* dst = result.ptr<float>(r);
                        for (int c = 0; c < min_dim; ++c) {
                            // Average the 3 channels
                            dst[c] += (src[c] + src[c + min_dim] + src[c + 2*min_dim]) / 3.0f;
                        }
                    }
                } else {
                    // Same dimension, just add
                    result += desc(cv::Range::all(), cv::Range(0, min_dim));
                }
            }

            // Average across components
            result /= static_cast<float>(component_descs.size());
            break;
        }
    }

    // L2 normalize the fused descriptors
    for (int i = 0; i < result.rows; ++i) {
        cv::Mat row = result.row(i);
        cv::normalize(row, row, 1.0, 0.0, cv::NORM_L2);
    }

    return result;
}

int PatchFusionExtractor::computeOutputDimension() const {
    switch (method_) {
        case PatchFusionMethod::CONCATENATE: {
            int total = 0;
            for (const auto& comp : components_) {
                total += comp->descriptorSize();
            }
            return total;
        }

        case PatchFusionMethod::CHANNEL_WISE: {
            // Output is the minimum dimension
            int min_dim = components_[0]->descriptorSize();
            for (const auto& comp : components_) {
                min_dim = std::min(min_dim, comp->descriptorSize());
            }
            return min_dim;
        }

        default:
            // All other methods preserve dimension of first component
            return components_[0]->descriptorSize();
    }
}

std::string PatchFusionExtractor::generateName() const {
    std::ostringstream oss;
    oss << "fusion_";

    for (size_t i = 0; i < components_.size(); ++i) {
        if (i > 0) oss << "+";
        oss << components_[i]->name();
    }

    oss << "__" << fusionMethodToString(method_);

    return oss.str();
}

bool PatchFusionExtractor::requiresResize() const {
    // If any component requires resize, we need to handle it
    for (const auto& comp : components_) {
        if (comp->requiresResize()) {
            return true;
        }
    }
    return false;
}

int PatchFusionExtractor::expectedPatchSize() const {
    // Return the maximum expected patch size among components
    // (PatchDescriptorFactory will provide appropriately sized patches)
    int max_size = 0;
    for (const auto& comp : components_) {
        max_size = std::max(max_size, comp->expectedPatchSize());
    }
    return max_size;
}

std::unique_ptr<IPatchDescriptorExtractor> PatchFusionExtractor::clone() const {
    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.reserve(components_.size());
    for (const auto& component : components_) {
        components.push_back(component->clone());
    }

    return std::make_unique<PatchFusionExtractor>(
        std::move(components),
        method_,
        weights_,
        name_);
}

} // namespace patches
} // namespace thesis_project
