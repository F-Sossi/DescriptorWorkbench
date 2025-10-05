#pragma once

#include "PoolingStrategy.hpp"
#include <memory>

namespace thesis_project::pooling {

/**
 * @brief Factory for creating pooling strategy instances
 * 
 * This factory creates the appropriate pooling strategy based on the
 * experiment configuration. It encapsulates the strategy selection logic
 * and makes it easy to add new pooling strategies in the future.
 */
class PoolingFactory {
public:
    /**
     * @brief Create a pooling strategy based on the experiment configuration
     * 
     * @param strategy The pooling strategy type from the configuration
     * @return PoolingStrategyPtr Unique pointer to the created strategy
     * @throws std::runtime_error If the strategy type is unknown
     */
    static PoolingStrategyPtr createStrategy(thesis_project::PoolingStrategy strategy);

    // New-config overload: create from descriptor configuration (Schema v1)
    static PoolingStrategyPtr createFromConfig(const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg);

    /**
     * @brief Get list of all available pooling strategy names
     * @return std::vector<std::string> List of strategy names for display/logging
     */
    static std::vector<std::string> getAvailableStrategies();

private:
    PoolingFactory() = default; // Static class, no instantiation
};

} // namespace thesis_project::pooling
