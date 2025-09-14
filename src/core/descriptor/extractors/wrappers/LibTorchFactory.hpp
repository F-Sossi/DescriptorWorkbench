// LibTorchFactory.hpp
#pragma once

#include <memory>
#include <string>

namespace thesis_project {
    class IDescriptorExtractor;

    namespace wrappers {
        // Factory functions to avoid header inclusion conflicts
        std::unique_ptr<IDescriptorExtractor> createLibTorchHardNet();
        std::unique_ptr<IDescriptorExtractor> createLibTorchSOSNet();
        std::unique_ptr<IDescriptorExtractor> createLibTorchL2Net();
    }
}