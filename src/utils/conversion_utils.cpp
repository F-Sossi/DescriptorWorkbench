#include "thesis_project/conversion_utils.hpp"

namespace thesis_project {
namespace conversion {

    PoolingStrategy toNewPoolingStrategy(int oldValue) {
        switch (oldValue) {
            case 0: return PoolingStrategy::NONE;               // NONE
            case 1: return PoolingStrategy::DOMAIN_SIZE_POOLING; // DOMAIN_SIZE_POOLING
            case 2: return PoolingStrategy::STACKING;           // STACKING
            default: return PoolingStrategy::NONE;
        }
    }

    int toOldPoolingStrategy(PoolingStrategy newValue) {
        switch (newValue) {
            case PoolingStrategy::NONE: return 0;
            case PoolingStrategy::DOMAIN_SIZE_POOLING: return 1;
            case PoolingStrategy::STACKING: return 2;
            default: return 0;
        }
    }

    DescriptorType toNewDescriptorType(int oldValue) {
        switch (oldValue) {
            case 0: return DescriptorType::SIFT;     // DESCRIPTOR_SIFT
            case 1: return DescriptorType::HoNC;     // DESCRIPTOR_HoNC
            case 2: return DescriptorType::RGBSIFT;  // DESCRIPTOR_RGBSIFT
            case 3: return DescriptorType::vSIFT;    // DESCRIPTOR_vSIFT
            case 4: return DescriptorType::NONE;     // NO_DESCRIPTOR
            default: return DescriptorType::SIFT;
        }
    }

    int toOldDescriptorType(DescriptorType newValue) {
        switch (newValue) {
            case DescriptorType::SIFT: return 0;
            case DescriptorType::HoNC: return 1;
            case DescriptorType::RGBSIFT: return 2;
            case DescriptorType::vSIFT: return 3;
            case DescriptorType::NONE: return 4;
            default: return 0;
        }
    }

    NormalizationStage toNewNormalizationStage(int oldValue) {
        switch (oldValue) {
            case 0: return NormalizationStage::BEFORE_POOLING;   // BEFORE_POOLING
            case 1: return NormalizationStage::AFTER_POOLING;    // AFTER_POOLING
            case 2: return NormalizationStage::NO_NORMALIZATION; // NO_NORMALIZATION
            default: return NormalizationStage::NO_NORMALIZATION;
        }
    }

    RootingStage toNewRootingStage(int oldValue) {
        switch (oldValue) {
            case 0: return RootingStage::R_BEFORE_POOLING; // R_BEFORE_POOLING
            case 1: return RootingStage::R_AFTER_POOLING;  // R_AFTER_POOLING
            case 2: return RootingStage::R_NONE;           // R_NONE
            default: return RootingStage::R_NONE;
        }
    }

} // namespace conversion
} // namespace thesis_project
