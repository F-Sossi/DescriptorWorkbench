// LibTorchFactory.cpp
#include "LibTorchFactory.hpp"
#include "LibTorchWrapper.hpp"

namespace thesis_project {
namespace wrappers {

std::unique_ptr<IDescriptorExtractor> createLibTorchHardNet() {
    return std::make_unique<LibTorchWrapper>("../models/hardnet.pt");
}

std::unique_ptr<IDescriptorExtractor> createLibTorchSOSNet() {
    return std::make_unique<LibTorchWrapper>("../models/sosnet.pt");
}

std::unique_ptr<IDescriptorExtractor> createLibTorchL2Net() {
    return std::make_unique<LibTorchWrapper>("../models/simple_l2net.pt");
}

} // namespace wrappers
} // namespace thesis_project