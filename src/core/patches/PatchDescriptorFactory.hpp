#pragma once

#include "PatchDescriptorExtractor.hpp"
#include <thesis_project/types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace thesis_project {
namespace patches {

/**
 * @brief Factory for creating patch descriptor extractors
 *
 * Creates appropriate extractors for:
 * - CNN descriptors (HardNet, SOSNet, L2Net)
 * - Traditional descriptors (SIFT, RGBSIFT, HoNC, DSPSIFT, SURF)
 * - Fusion descriptors (combinations with various fusion methods)
 */
class PatchDescriptorFactory {
public:
    /**
     * @brief Create a single patch descriptor extractor by type
     * @param type Descriptor type enum
     * @return Unique pointer to the extractor
     */
    static std::unique_ptr<IPatchDescriptorExtractor> create(DescriptorType type);

    /**
     * @brief Create a patch descriptor extractor by type name string
     * @param type_name String name (e.g., "sift", "hardnet", "libtorch_sosnet")
     * @return Unique pointer to the extractor
     */
    static std::unique_ptr<IPatchDescriptorExtractor> create(const std::string& type_name);

    /**
     * @brief Create a fusion extractor from component types
     * @param component_types Vector of descriptor types to fuse
     * @param method Fusion method
     * @param weights Optional weights for weighted average
     * @param name Custom name for the fusion descriptor
     * @return Unique pointer to the fusion extractor
     */
    static std::unique_ptr<IPatchDescriptorExtractor> createFusion(
        const std::vector<DescriptorType>& component_types,
        PatchFusionMethod method = PatchFusionMethod::CONCATENATE,
        const std::vector<float>& weights = {},
        const std::string& name = "");

    /**
     * @brief Create a fusion extractor from component type name strings
     * @param component_names Vector of descriptor type names
     * @param method_name Fusion method name string
     * @param weights Optional weights
     * @param name Custom name
     * @return Unique pointer to the fusion extractor
     */
    static std::unique_ptr<IPatchDescriptorExtractor> createFusion(
        const std::vector<std::string>& component_names,
        const std::string& method_name = "concatenate",
        const std::vector<float>& weights = {},
        const std::string& name = "");

    /**
     * @brief Check if a descriptor type is supported for patches
     * @param type Descriptor type
     * @return true if supported
     */
    static bool isSupported(DescriptorType type);

    /**
     * @brief Get list of all supported descriptor type names
     * @return Vector of supported type name strings
     */
    static std::vector<std::string> supportedTypes();

private:
    /**
     * @brief Convert string to DescriptorType
     */
    static DescriptorType stringToType(const std::string& name);
};

} // namespace patches
} // namespace thesis_project
