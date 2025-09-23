#include "MatchingFactory.hpp"
#include "BruteForceMatching.hpp"
#include "RatioTestMatching.hpp"
#include <stdexcept>

namespace thesis_project::matching {

MatchingStrategyPtr MatchingFactory::createStrategy(::MatchingStrategy strategy) {
    switch (strategy) {
        case BRUTE_FORCE:
            return std::make_unique<BruteForceMatching>();
            
        case FLANN:
            throw std::runtime_error("FLANN matching strategy not yet implemented");
            
        case RATIO_TEST:
            return std::make_unique<RatioTestMatching>();
            
        default:
            throw std::runtime_error("Unknown matching strategy: " + std::to_string(static_cast<int>(strategy)));
    }
}

MatchingStrategyPtr MatchingFactory::createFromConfig(const experiment_config& config) {
    return createStrategy(config.matchingStrategy);
}

std::vector<std::string> MatchingFactory::getAvailableStrategies() {
    return {
        "BruteForce",
        "FLANN (future)",
        "RatioTest"
    };
}

} // namespace thesis_project::matching