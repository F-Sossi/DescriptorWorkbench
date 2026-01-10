#include "PatchDescriptorFactory.hpp"
#include "PatchCNNExtractor.hpp"
#include "PatchTraditionalExtractor.hpp"
#include "PatchFusionExtractor.hpp"
#include <algorithm>
#include <stdexcept>

namespace thesis_project {
namespace patches {

std::unique_ptr<IPatchDescriptorExtractor> PatchDescriptorFactory::create(DescriptorType type) {
    switch (type) {
        // CNN descriptors
        case DescriptorType::LIBTORCH_HARDNET:
            return createPatchHardNet();

        case DescriptorType::LIBTORCH_SOSNET:
            return createPatchSOSNet();

        case DescriptorType::LIBTORCH_L2NET:
            return createPatchL2Net();

        // Traditional descriptors
        case DescriptorType::SIFT:
            return createPatchSIFT();

        case DescriptorType::RGBSIFT:
            return createPatchRGBSIFT();

        case DescriptorType::RGBSIFT_CHANNEL_AVG:
            return createPatchRGBSIFTChannelAvg();

        case DescriptorType::HoNC:
            return createPatchHoNC();

        case DescriptorType::DSPSIFT_V2:
        case DescriptorType::DSPSIFT:
            return createPatchDSPSIFT();

        case DescriptorType::SURF:
            return createPatchSURF();

        default:
            throw std::invalid_argument("PatchDescriptorFactory: unsupported descriptor type");
    }
}

std::unique_ptr<IPatchDescriptorExtractor> PatchDescriptorFactory::create(const std::string& type_name) {
    return create(stringToType(type_name));
}

std::unique_ptr<IPatchDescriptorExtractor> PatchDescriptorFactory::createFusion(
    const std::vector<DescriptorType>& component_types,
    PatchFusionMethod method,
    const std::vector<float>& weights,
    const std::string& name) {

    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.reserve(component_types.size());

    for (auto type : component_types) {
        components.push_back(create(type));
    }

    return std::make_unique<PatchFusionExtractor>(
        std::move(components), method, weights, name);
}

std::unique_ptr<IPatchDescriptorExtractor> PatchDescriptorFactory::createFusion(
    const std::vector<std::string>& component_names,
    const std::string& method_name,
    const std::vector<float>& weights,
    const std::string& name) {

    std::vector<std::unique_ptr<IPatchDescriptorExtractor>> components;
    components.reserve(component_names.size());

    for (const auto& comp_name : component_names) {
        components.push_back(create(comp_name));
    }

    PatchFusionMethod method = stringToFusionMethod(method_name);

    return std::make_unique<PatchFusionExtractor>(
        std::move(components), method, weights, name);
}

bool PatchDescriptorFactory::isSupported(DescriptorType type) {
    switch (type) {
        case DescriptorType::SIFT:
        case DescriptorType::RGBSIFT:
        case DescriptorType::RGBSIFT_CHANNEL_AVG:
        case DescriptorType::HoNC:
        case DescriptorType::DSPSIFT:
        case DescriptorType::DSPSIFT_V2:
        case DescriptorType::SURF:
        case DescriptorType::LIBTORCH_HARDNET:
        case DescriptorType::LIBTORCH_SOSNET:
        case DescriptorType::LIBTORCH_L2NET:
            return true;
        default:
            return false;
    }
}

std::vector<std::string> PatchDescriptorFactory::supportedTypes() {
    return {
        "sift",
        "rgbsift",
        "rgbsift_channel_avg",
        "honc",
        "dspsift",
        "dspsift_v2",
        "surf",
        "hardnet",
        "libtorch_hardnet",
        "sosnet",
        "libtorch_sosnet",
        "l2net",
        "libtorch_l2net"
    };
}

DescriptorType PatchDescriptorFactory::stringToType(const std::string& name) {
    // Convert to lowercase for comparison
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    // CNN descriptors
    if (lower_name == "hardnet" || lower_name == "libtorch_hardnet") {
        return DescriptorType::LIBTORCH_HARDNET;
    }
    if (lower_name == "sosnet" || lower_name == "libtorch_sosnet") {
        return DescriptorType::LIBTORCH_SOSNET;
    }
    if (lower_name == "l2net" || lower_name == "libtorch_l2net") {
        return DescriptorType::LIBTORCH_L2NET;
    }

    // Traditional descriptors
    if (lower_name == "sift") {
        return DescriptorType::SIFT;
    }
    if (lower_name == "rgbsift") {
        return DescriptorType::RGBSIFT;
    }
    if (lower_name == "rgbsift_channel_avg" || lower_name == "rgbsift_avg") {
        return DescriptorType::RGBSIFT_CHANNEL_AVG;
    }
    if (lower_name == "honc") {
        return DescriptorType::HoNC;
    }
    if (lower_name == "dspsift" || lower_name == "dspsift_v2") {
        return DescriptorType::DSPSIFT_V2;
    }
    if (lower_name == "surf") {
        return DescriptorType::SURF;
    }

    throw std::invalid_argument("PatchDescriptorFactory: unknown descriptor type: " + name);
}

} // namespace patches
} // namespace thesis_project
