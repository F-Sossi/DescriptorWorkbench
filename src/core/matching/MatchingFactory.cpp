#include "MatchingFactory.hpp"
#include "BruteForceMatching.hpp"
#include "RatioTestMatching.hpp"
#include "FLANNMatching.hpp"
#include <stdexcept>

namespace thesis_project::matching {

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
