#include "PoolingFactory.hpp"
#include "NoPooling.hpp"
#include "DomainSizePooling.hpp"
#include "StackingPooling.hpp"
#include <stdexcept>

namespace thesis_project::pooling {

PoolingStrategyPtr PoolingFactory::createStrategy(thesis_project::PoolingStrategy strategy) {
    using thesis_project::PoolingStrategy;
    switch (strategy) {
        case PoolingStrategy::NONE:
            return std::make_unique<NoPooling>();
        case PoolingStrategy::DOMAIN_SIZE_POOLING:
            return std::make_unique<DomainSizePooling>();
        case PoolingStrategy::STACKING:
            return std::make_unique<StackingPooling>();
        default:
            throw std::runtime_error("Unknown pooling strategy: " + std::to_string(static_cast<int>(strategy)));
    }
}

PoolingStrategyPtr PoolingFactory::createFromConfig(const thesis_project::config::ExperimentConfig::DescriptorConfig& descCfg) {
    using thesis_project::PoolingStrategy;
    switch (descCfg.params.pooling) {
        case PoolingStrategy::NONE: return std::make_unique<NoPooling>();
        case PoolingStrategy::DOMAIN_SIZE_POOLING: return std::make_unique<DomainSizePooling>();
        case PoolingStrategy::STACKING: return std::make_unique<StackingPooling>();
        default: throw std::runtime_error("Unknown pooling strategy (Schema v1)");
    }
}

std::vector<std::string> PoolingFactory::getAvailableStrategies() {
    return {
        "None",
        "DomainSizePooling", 
        "Stacking"
    };
}

} // namespace thesis_project::pooling
