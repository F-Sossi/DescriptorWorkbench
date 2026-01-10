#pragma once

#include "PatchDescriptorExtractor.hpp"
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <string>

namespace thesis_project {
namespace patches {

/**
 * @brief Combines multiple patch descriptor extractors using various fusion strategies
 *
 * Supports all fusion methods:
 * - AVERAGE: Element-wise average (requires matching dimensions)
 * - WEIGHTED_AVG: Weighted average with configurable weights
 * - MAX: Element-wise maximum
 * - MIN: Element-wise minimum
 * - CONCATENATE: Horizontal concatenation (dimensions add up)
 * - CHANNEL_WISE: Special fusion for 128D + 384D -> 128D or 384D
 */
class PatchFusionExtractor : public IPatchDescriptorExtractor {
public:
    /**
     * @brief Construct a fusion extractor
     * @param components Vector of component extractors (takes ownership)
     * @param method Fusion method to apply
     * @param weights Optional weights for WEIGHTED_AVG (must match component count)
     * @param name_override Optional custom name (otherwise auto-generated)
     */
    explicit PatchFusionExtractor(
        std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components,
        PatchFusionMethod method = PatchFusionMethod::CONCATENATE,
        const std::vector<float>& weights = {},
        const std::string& name_override = "");

    ~PatchFusionExtractor() override = default;

    /**
     * @brief Extract fused descriptors from patches
     *
     * 1. Run each component extractor on the patches
     * 2. Apply the fusion method to combine results
     *
     * @param patches Vector of patches
     * @param params Descriptor parameters
     * @return Fused descriptors
     */
    cv::Mat extractFromPatches(
        const std::vector<cv::Mat>& patches,
        const DescriptorParams& params) override;

    std::string name() const override { return name_; }
    int descriptorSize() const override { return output_dim_; }
    int descriptorType() const override { return CV_32F; }
    bool requiresResize() const override;
    int expectedPatchSize() const override;
    std::unique_ptr<IPatchDescriptorExtractor> clone() const override;

    /**
     * @brief Get the number of component extractors
     */
    size_t numComponents() const { return components_.size(); }

    /**
     * @brief Get the fusion method
     */
    PatchFusionMethod fusionMethod() const { return method_; }

private:
    /**
     * @brief Apply the fusion method to component descriptors
     */
    cv::Mat fuseDescriptors(const std::vector<cv::Mat>& component_descs) const;

    /**
     * @brief Compute the output dimension based on components and fusion method
     */
    int computeOutputDimension() const;

    /**
     * @brief Generate a default name based on components
     */
    std::string generateName() const;

    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components_;
    PatchFusionMethod method_;
    std::vector<float> weights_;
    std::string name_;
    int output_dim_;
};

} // namespace patches
} // namespace thesis_project
