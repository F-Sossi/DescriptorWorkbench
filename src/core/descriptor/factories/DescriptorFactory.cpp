#include "DescriptorFactory.hpp"
#include "src/core/descriptor/extractors/wrappers/SIFTWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/RGBSIFTWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/HoNCWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/VSIFTWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPSIFTWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPSIFTWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.hpp"
#include "src/core/descriptor/extractors/wrappers/DNNPatchWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/VGGWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/ORBWrapper.hpp"
#include "src/core/descriptor/extractors/wrappers/SURFWrapper.hpp"
#ifdef BUILD_LIBTORCH_DESCRIPTORS
#include "../extractors/wrappers/LibTorchFactory.hpp"
#endif
#include <stdexcept>

namespace thesis_project::factories {

std::vector<std::string> DescriptorFactory::getSupportedTypes() {
    std::vector<std::string> types = {"SIFT", "RGBSIFT", "HoNC", "VSIFT", "DSPSIFT"};
#ifdef HAVE_OPENCV_XFEATURES2D
    types.emplace_back("VGG");
#endif
    return types;
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
        case thesis_project::DescriptorType::DSPSIFT_V2:
            return std::make_unique<thesis_project::DSPSIFTWrapperV2>();
        case thesis_project::DescriptorType::DSPRGBSIFT_V2:
            return std::make_unique<wrappers::DSPRGBSIFTWrapperV2>();
        case thesis_project::DescriptorType::DSPHOWH_V2:
            return std::make_unique<wrappers::DSPHoWHWrapperV2>();
        case thesis_project::DescriptorType::DSPHONC_V2:
            return std::make_unique<wrappers::DSPHoNCWrapperV2>();
        case thesis_project::DescriptorType::VGG:
            return std::make_unique<wrappers::VGGWrapper>();
        case thesis_project::DescriptorType::ORB:
            return std::make_unique<wrappers::ORBWrapper>();
        case thesis_project::DescriptorType::SURF:
            return std::make_unique<wrappers::SURFWrapper>();
#ifdef BUILD_LIBTORCH_DESCRIPTORS
        case thesis_project::DescriptorType::LIBTORCH_HARDNET:
            return wrappers::createLibTorchHardNet();
        case thesis_project::DescriptorType::LIBTORCH_SOSNET:
            return wrappers::createLibTorchSOSNet();
        case thesis_project::DescriptorType::LIBTORCH_L2NET:
            return wrappers::createLibTorchL2Net();
#endif
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
        case thesis_project::DescriptorType::DSPSIFT_V2:
        case thesis_project::DescriptorType::DSPRGBSIFT_V2:
        case thesis_project::DescriptorType::DSPHOWH_V2:
        case thesis_project::DescriptorType::DSPHONC_V2:
        case thesis_project::DescriptorType::VGG:
        case thesis_project::DescriptorType::ORB:
        case thesis_project::DescriptorType::SURF:
#ifdef BUILD_LIBTORCH_DESCRIPTORS
        case thesis_project::DescriptorType::LIBTORCH_HARDNET:
        case thesis_project::DescriptorType::LIBTORCH_SOSNET:
        case thesis_project::DescriptorType::LIBTORCH_L2NET:
#endif
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

} // namespace thesis_project::factories
