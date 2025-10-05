#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include <memory>
#include <vector>
#include <string>

namespace thesis_project {
namespace factories {

/**
 * @brief Factory for creating descriptor extractors from experiment configuration
 */
class DescriptorFactory {
public:
    static std::unique_ptr<IDescriptorExtractor> create(thesis_project::DescriptorType type);
    static bool isSupported(thesis_project::DescriptorType type);
    static std::vector<std::string> getSupportedTypes();

private:
    static std::unique_ptr<IDescriptorExtractor> createSIFT();
    static std::unique_ptr<IDescriptorExtractor> createRGBSIFT();
};

} // namespace factories
} // namespace thesis_project
