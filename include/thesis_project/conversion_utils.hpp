#pragma once

#include "thesis_project/types.hpp"

/**
 * @file conversion_utils.hpp
 * @brief Utilities for converting between old and new type systems
 *
 * This file provides conversion functions between the old enum system
 * and the new scoped enum system during the migration period.
 */

namespace thesis_project {
namespace conversion {

    /**
     * @brief Convert old-style pooling strategy to new scoped enum
     */
    PoolingStrategy toNewPoolingStrategy(int oldValue);

    /**
     * @brief Convert new scoped enum to old-style value
     */
    int toOldPoolingStrategy(PoolingStrategy newValue);

    /**
     * @brief Convert old-style descriptor type to new scoped enum
     */
    DescriptorType toNewDescriptorType(int oldValue);

    /**
     * @brief Convert new scoped enum to old-style value
     */
    int toOldDescriptorType(DescriptorType newValue);

    /**
     * @brief Convert old-style normalization stage to new scoped enum
     */
    NormalizationStage toNewNormalizationStage(int oldValue);

    /**
     * @brief Convert old-style rooting stage to new scoped enum
     */
    RootingStage toNewRootingStage(int oldValue);

} // namespace conversion
} // namespace thesis_project
