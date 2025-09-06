#pragma once

#include "interfaces/IDescriptorExtractor.hpp"
#include "src/core/config/experiment_config.hpp"
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
    static std::unique_ptr<IDescriptorExtractor> create(const experiment_config& config);
    static std::unique_ptr<IDescriptorExtractor> tryCreate(const experiment_config& config) noexcept;
    static bool isSupported(const experiment_config& config);
    static std::vector<std::string> getSupportedTypes();

    // New-config overloads (Schema v1)
    static std::unique_ptr<IDescriptorExtractor> create(thesis_project::DescriptorType type);
    static bool isSupported(thesis_project::DescriptorType type);

private:
    static std::unique_ptr<IDescriptorExtractor> createSIFT(const experiment_config& config);
    static std::unique_ptr<IDescriptorExtractor> createRGBSIFT(const experiment_config& config);
    static std::unique_ptr<IDescriptorExtractor> createSIFT();
    static std::unique_ptr<IDescriptorExtractor> createRGBSIFT();
};

} // namespace factories
} // namespace thesis_project
