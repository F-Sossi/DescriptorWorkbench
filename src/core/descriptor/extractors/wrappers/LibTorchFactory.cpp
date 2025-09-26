// LibTorchFactory.cpp
#include "LibTorchFactory.hpp"
#include "LibTorchWrapper.hpp"

namespace thesis_project {
namespace wrappers {

std::unique_ptr<IDescriptorExtractor> createLibTorchHardNet() {
    // Use Kornia-compatible settings: per-patch standardization enabled, proper scale multiplier
    return std::make_unique<LibTorchWrapper>("../models/hardnet.pt", 32, 2.0f, true, 0.0f, 1.0f, true, 128, cv::INTER_LINEAR);
}

std::unique_ptr<IDescriptorExtractor> createLibTorchSOSNet() {
    // Use same Kornia-compatible settings as HardNet: per-patch standardization enabled
    return std::make_unique<LibTorchWrapper>("../models/sosnet.pt", 32, 2.0f, true, 0.0f, 1.0f, true, 128, cv::INTER_LINEAR);
}

std::unique_ptr<IDescriptorExtractor> createLibTorchL2Net() {
    return std::make_unique<LibTorchWrapper>("../models/simple_l2net.pt");
}

} // namespace wrappers
} // namespace thesis_project