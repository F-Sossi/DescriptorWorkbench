#include "DescriptorFactory.hpp"
#include "../extractors/wrappers/SIFTWrapper.hpp"
#include "../extractors/wrappers/RGBSIFTWrapper.hpp"
#include "../extractors/wrappers/HoNCWrapper.hpp"
#include "../extractors/wrappers/VSIFTWrapper.hpp"
#include "../extractors/wrappers/DSPSIFTWrapper.hpp"
#include "../extractors/wrappers/DNNPatchWrapper.hpp"
#include "../extractors/wrappers/PseudoDNNWrapper.hpp"
#include "../extractors/wrappers/VGGWrapper.hpp"
#include <stdexcept>

namespace thesis_project {
namespace factories {

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::create(const experiment_config& config) {
    switch (config.descriptorOptions.descriptorType) {
        case DESCRIPTOR_SIFT:
            return createSIFT(config);
        case DESCRIPTOR_RGBSIFT:
            return createRGBSIFT(config);
        case DESCRIPTOR_HoNC:
            return std::make_unique<wrappers::HoNCWrapper>(config);
        case DESCRIPTOR_vSIFT:
            return std::make_unique<wrappers::VSIFTWrapper>(config);
        // DSPSIFT maps to new interface only (legacy maps as SIFT)
        default:
            throw std::runtime_error("Unsupported descriptor type in factory");
    }
}

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::tryCreate(const experiment_config& config) noexcept {
    try {
        return create(config);
    } catch (...) {
        return nullptr;
    }
}

bool DescriptorFactory::isSupported(const experiment_config& config) {
    switch (config.descriptorOptions.descriptorType) {
        case DESCRIPTOR_SIFT:
        case DESCRIPTOR_RGBSIFT:
        case DESCRIPTOR_HoNC:
        case DESCRIPTOR_vSIFT:
            // DSPSIFT not exposed via legacy enum; use new-config path
            return true;
        default:
            return false;
    }
}

std::vector<std::string> DescriptorFactory::getSupportedTypes() {
    std::vector<std::string> types = {"SIFT", "RGBSIFT", "HoNC", "VSIFT", "DSPSIFT"};
#ifdef HAVE_OPENCV_XFEATURES2D
    types.push_back("VGG");
#endif
    return types;
}

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::createSIFT(const experiment_config& config) {
    return std::make_unique<wrappers::SIFTWrapper>(config);
}

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::createRGBSIFT(const experiment_config& config) {
    return std::make_unique<wrappers::RGBSIFTWrapper>(config);
}

// New-config overloads
std::unique_ptr<IDescriptorExtractor> DescriptorFactory::create(thesis_project::DescriptorType type) {
    switch (type) {
        case thesis_project::DescriptorType::SIFT:
            return createSIFT();
        case thesis_project::DescriptorType::RGBSIFT:
            return createRGBSIFT();
        case thesis_project::DescriptorType::HoNC:
            return std::make_unique<wrappers::HoNCWrapper>();
        case thesis_project::DescriptorType::vSIFT:
            return std::make_unique<wrappers::VSIFTWrapper>();
        case thesis_project::DescriptorType::DSPSIFT:
            return std::make_unique<wrappers::DSPSIFTWrapper>();
        case thesis_project::DescriptorType::VGG:
            return std::make_unique<wrappers::VGGWrapper>();
        // DNNPatch created via DescriptorConfig params (model path required) elsewhere
        default:
            throw std::runtime_error("Unsupported descriptor type in factory (new-config)");
    }
}

bool DescriptorFactory::isSupported(thesis_project::DescriptorType type) {
    switch (type) {
        case thesis_project::DescriptorType::SIFT:
        case thesis_project::DescriptorType::RGBSIFT:
        case thesis_project::DescriptorType::HoNC:
        case thesis_project::DescriptorType::vSIFT:
        case thesis_project::DescriptorType::DSPSIFT:
        case thesis_project::DescriptorType::VGG:
            return true;
        default:
            return false;
    }
}

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::createSIFT() {
    return std::make_unique<wrappers::SIFTWrapper>();
}

std::unique_ptr<IDescriptorExtractor> DescriptorFactory::createRGBSIFT() {
    return std::make_unique<wrappers::RGBSIFTWrapper>();
}

} // namespace factories
} // namespace thesis_project
