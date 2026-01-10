#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <thesis_project/types.hpp>

namespace thesis_project {
namespace patches {

/**
 * @brief Interface for extracting descriptors directly from pre-extracted patches
 *
 * Unlike IDescriptorExtractor which takes full images + keypoints,
 * this interface takes already-extracted patches and computes descriptors.
 * This is used for the HPatches patch benchmark where patches are pre-extracted.
 */
class IPatchDescriptorExtractor {
public:
    virtual ~IPatchDescriptorExtractor() = default;

    /**
     * @brief Extract descriptors from a batch of patches
     * @param patches Vector of patches (typically 65x65 or 32x32 grayscale)
     * @param params Descriptor parameters (device, etc.)
     * @return cv::Mat where each row is a descriptor for the corresponding patch
     */
    virtual cv::Mat extractFromPatches(
        const std::vector<cv::Mat>& patches,
        const DescriptorParams& params) = 0;

    /**
     * @brief Get the descriptor name
     * @return Human-readable descriptor name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Get the descriptor dimension
     * @return Number of elements per descriptor (e.g., 128 for SIFT/HardNet)
     */
    virtual int descriptorSize() const = 0;

    /**
     * @brief Get the OpenCV type of descriptor elements
     * @return CV_32F for most descriptors
     */
    virtual int descriptorType() const { return CV_32F; }

    /**
     * @brief Check if this extractor requires 32x32 input (CNN) or can use 65x65 (traditional)
     * @return true if patches should be resized to 32x32 before extraction
     */
    virtual bool requiresResize() const { return true; }

    /**
     * @brief Get the expected input patch size
     * @return Expected patch size (32 for CNN, 65 for traditional that handle their own sizing)
     */
    virtual int expectedPatchSize() const { return 32; }

    /**
     * @brief Create a copy of this extractor (used for per-thread instances)
     * @return New extractor instance with the same configuration
     */
    virtual std::unique_ptr<IPatchDescriptorExtractor> clone() const = 0;
};

/**
 * @brief Enumeration of fusion methods for combining descriptors
 */
enum class PatchFusionMethod {
    AVERAGE,        ///< Element-wise average
    WEIGHTED_AVG,   ///< Weighted average with configurable weights
    MAX,            ///< Element-wise maximum
    MIN,            ///< Element-wise minimum
    CONCATENATE,    ///< Horizontal concatenation (increases dimension)
    CHANNEL_WISE    ///< Channel-wise fusion for compatible dimensions
};

/**
 * @brief Convert string to PatchFusionMethod
 */
inline PatchFusionMethod stringToFusionMethod(const std::string& method) {
    if (method == "average" || method == "avg") return PatchFusionMethod::AVERAGE;
    if (method == "weighted_avg" || method == "weighted") return PatchFusionMethod::WEIGHTED_AVG;
    if (method == "max") return PatchFusionMethod::MAX;
    if (method == "min") return PatchFusionMethod::MIN;
    if (method == "concatenate" || method == "concat") return PatchFusionMethod::CONCATENATE;
    if (method == "channel_wise" || method == "channelwise") return PatchFusionMethod::CHANNEL_WISE;
    throw std::invalid_argument("Unknown fusion method: " + method);
}

/**
 * @brief Convert PatchFusionMethod to string
 */
inline std::string fusionMethodToString(PatchFusionMethod method) {
    switch (method) {
        case PatchFusionMethod::AVERAGE: return "average";
        case PatchFusionMethod::WEIGHTED_AVG: return "weighted_avg";
        case PatchFusionMethod::MAX: return "max";
        case PatchFusionMethod::MIN: return "min";
        case PatchFusionMethod::CONCATENATE: return "concatenate";
        case PatchFusionMethod::CHANNEL_WISE: return "channel_wise";
        default: return "unknown";
    }
}

} // namespace patches
} // namespace thesis_project
