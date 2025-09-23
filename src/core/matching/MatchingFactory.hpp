#pragma once

#include "MatchingStrategy.hpp"
#include "src/core/config/experiment_config.hpp"
#include <memory>
#include <vector>
#include <string>

namespace thesis_project::matching {

/**
 * @brief Factory for creating matching strategy instances
 * 
 * This factory creates the appropriate matching strategy based on the
 * experiment configuration. It encapsulates the strategy selection logic
 * and makes it easy to add new matching strategies in the future.
 * 
 * Supported strategies:
 * - BruteForce: Simple brute-force matching with cross-check
 * - RatioTest: Lowe's ratio test (SNN) for robust matching
 *
 * Future strategies could include:
 * - FLANN: Fast approximate matching
 * - Hybrid: Combining multiple strategies
 */
class MatchingFactory {
public:
    /**
     * @brief Create a matching strategy based on the experiment configuration
     * 
     * @param strategy The matching strategy type from the configuration
     * @return MatchingStrategyPtr Unique pointer to the created strategy
     * @throws std::runtime_error If the strategy type is unknown
     */
    static MatchingStrategyPtr createStrategy(::MatchingStrategy strategy);
    
    /**
     * @brief Create a matching strategy from experiment config
     * 
     * Convenience method that extracts the matching strategy from the
     * experiment configuration and creates the appropriate instance.
     * 
     * @param config Experiment configuration containing matching strategy
     * @return MatchingStrategyPtr Unique pointer to the created strategy
     */
    static MatchingStrategyPtr createFromConfig(const experiment_config& config);

    /**
     * @brief Get list of all available matching strategy names
     * @return std::vector<std::string> List of strategy names for display/logging
     */
    static std::vector<std::string> getAvailableStrategies();

private:
    MatchingFactory() = default; // Static class, no instantiation
};

} // namespace thesis_project::matching