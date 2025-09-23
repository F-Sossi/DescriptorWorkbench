// LibTorchFactory.cpp
#include "LibTorchFactory.hpp"
#include "LibTorchWrapper.hpp"

namespace thesis_project {
namespace wrappers {

std::unique_ptr<IDescriptorExtractor> createLibTorchHardNet() {
    // Test: Disable per-patch standardization, use simple [0,1] normalization
    return std::make_unique<LibTorchWrapper>("../models/hardnet.pt", 32, 5.0f, true, 0.0f, 1.0f, false, 128, cv::INTER_AREA);
}

std::unique_ptr<IDescriptorExtractor> createLibTorchSOSNet() {
    return std::make_unique<LibTorchWrapper>("../models/sosnet.pt");
}

std::unique_ptr<IDescriptorExtractor> createLibTorchL2Net() {
    return std::make_unique<LibTorchWrapper>("../models/simple_l2net.pt");
}

} // namespace wrappers
} // namespace thesis_project