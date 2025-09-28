#include "MatchingFactory.hpp"
#include "BruteForceMatching.hpp"
#include "RatioTestMatching.hpp"
#include "FLANNMatching.hpp"
#include <stdexcept>

namespace thesis_project::matching {

MatchingStrategyPtr MatchingFactory::createStrategy(::MatchingStrategy strategy) {
    switch (strategy) {
        case BRUTE_FORCE:
            return std::make_unique<BruteForceMatching>();
            
        case FLANN:
            return std::make_unique<FLANNMatching>();
            
        case RATIO_TEST:
            return std::make_unique<RatioTestMatching>();
            
        default:
            throw std::runtime_error("Unknown matching strategy: " + std::to_string(static_cast<int>(strategy)));
    }
}

MatchingStrategyPtr MatchingFactory::createFromConfig(const experiment_config& config) {
    return createStrategy(config.matchingStrategy);
}

MatchingStrategyPtr MatchingFactory::createStrategy(thesis_project::MatchingMethod method) {
    switch (method) {
        case thesis_project::MatchingMethod::BRUTE_FORCE:
            return std::make_unique<BruteForceMatching>();

        case thesis_project::MatchingMethod::FLANN:
            return std::make_unique<FLANNMatching>();

        case thesis_project::MatchingMethod::RATIO_TEST:
            return std::make_unique<RatioTestMatching>();

        default:
            throw std::runtime_error("Unknown matching method: " + std::to_string(static_cast<int>(method)));
    }
}

std::vector<std::string> MatchingFactory::getAvailableStrategies() {
    return {
        "BruteForce",
        "FLANN",
        "RatioTest"
    };
}

} // namespace thesis_project::matching