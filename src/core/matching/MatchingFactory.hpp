#pragma once

#include "MatchingStrategy.hpp"
#include "include/thesis_project/types.hpp"
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
     * @brief Create a matching strategy based on the new MatchingMethod enum
     *
     * @param method The matching method type from the new enum system
     * @return MatchingStrategyPtr Unique pointer to the created strategy
     * @throws std::runtime_error If the method type is unknown
     */
    static MatchingStrategyPtr createStrategy(thesis_project::MatchingMethod method);
    /**
     * @brief Get list of all available matching strategy names
     * @return std::vector<std::string> List of strategy names for display/logging
     */
    static std::vector<std::string> getAvailableStrategies();

private:
    MatchingFactory() = default; // Static class, no instantiation
};

} // namespace thesis_project::matching
