#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace thesis_project {

    // Forward declarations
    class IKeypointGenerator;
    class IDescriptorExtractor;
    class IEvaluator;

    // ================================
    // EXTRACTED TYPES FROM experiment_config.hpp
    // ================================

    /**
     * @brief Modern C++17 scoped enum for pooling strategies
     */
    enum class PoolingStrategy {
        NONE,                   ///< No pooling
        DOMAIN_SIZE_POOLING,   ///< Domain size pooling
        STACKING               ///< Stacking pooling
    };

    /**
     * @brief When to apply normalization during processing
     */
    enum class NormalizationStage {
        BEFORE_POOLING,        ///< Normalization before pooling
        AFTER_POOLING,         ///< Normalization after pooling
        NO_NORMALIZATION       ///< Skip normalization
    };

    /**
     * @brief When to apply rooting during processing
     */
    enum class RootingStage {
        R_BEFORE_POOLING,      ///< Rooting before pooling
        R_AFTER_POOLING,       ///< Rooting after pooling
        R_NONE                 ///< No rooting
    };

    /**
     * @brief Descriptor types available in the system
     */
    enum class DescriptorType {
        SIFT,                  ///< Standard SIFT descriptor
        HoNC,                  ///< Histogram of Normalized Colors
        RGBSIFT,               ///< RGB color SIFT
        vSIFT,                 ///< Vanilla SIFT implementation
        DSPSIFT,               ///< Domain-Size Pooled SIFT (professor's implementation)
        VGG,                   ///< VGG descriptor from OpenCV xfeatures2d (non-pooled)
        DNN_PATCH,             ///< ONNX-backed patch descriptor via cv::dnn
        LIBTORCH_HARDNET,      ///< LibTorch HardNet CNN descriptor
        LIBTORCH_SOSNET,       ///< LibTorch SOSNet CNN descriptor
        LIBTORCH_L2NET,        ///< LibTorch L2-Net CNN descriptor
        NONE                   ///< No descriptor
    };

    /**
     * @brief Color space for descriptor computation
     */
    enum class DescriptorColorSpace {
        COLOR,                 ///< Color descriptor
        BW                     ///< Black and white descriptor
    };

    /**
     * @brief Image processing color mode
     */
    enum class ImageType {
        COLOR,                 ///< Color image processing
        BW                     ///< Black and white image processing
    };

    /**
     * @brief Visual verification options for debugging
     */
    enum class VerificationType {
        MATCHES,               ///< Verification using descriptor matches
        HOMOGRAPHY,           ///< Verification using homography projection
        NO_VISUAL_VERIFICATION ///< No visual verification
    };

    /**
     * @brief Keypoint generation methods
     */
    enum class KeypointGenerator {
        SIFT,
        HARRIS,
        ORB,
        LOCKED_IN  // For using pre-computed keypoints
    };

    /**
     * @brief Keypoint source strategies for evaluation
     */
    enum class KeypointSource {
        HOMOGRAPHY_PROJECTION,  ///< Transform keypoints from reference using homography (controlled evaluation)
        INDEPENDENT_DETECTION   ///< Detect keypoints fresh on each image (realistic evaluation)
    };

    /**
     * @brief Matching algorithms
     */
    enum class MatchingMethod {
        BRUTE_FORCE,
        FLANN
    };

    /**
     * @brief Validation methods for match quality
     */
    enum class ValidationMethod {
        HOMOGRAPHY,
        CROSS_IMAGE,
        NONE
    };

    // ================================
    // CONVERSION FUNCTIONS FOR COMPATIBILITY
    // ================================

    /**
     * @brief Convert old-style enum to new scoped enum
     */
    inline PoolingStrategy toNewPoolingStrategy(int oldValue) {
        switch (oldValue) {
            case 0: return PoolingStrategy::NONE;
            case 1: return PoolingStrategy::DOMAIN_SIZE_POOLING;
            case 2: return PoolingStrategy::STACKING;
            default: return PoolingStrategy::NONE;
        }
    }

    /**
     * @brief Convert new scoped enum to old-style enum value
     */
    inline int toOldPoolingStrategy(PoolingStrategy newValue) {
        switch (newValue) {
            case PoolingStrategy::NONE: return 0;
            case PoolingStrategy::DOMAIN_SIZE_POOLING: return 1;
            case PoolingStrategy::STACKING: return 2;
            default: return 0;
        }
    }

    inline DescriptorType toNewDescriptorType(int oldValue) {
        switch (oldValue) {
            case 0: return DescriptorType::SIFT;      // DESCRIPTOR_SIFT
            case 1: return DescriptorType::HoNC;      // DESCRIPTOR_HoNC
            case 2: return DescriptorType::RGBSIFT;   // DESCRIPTOR_RGBSIFT
            case 3: return DescriptorType::vSIFT;     // DESCRIPTOR_vSIFT
            case 4: return DescriptorType::NONE;      // NO_DESCRIPTOR
            default: return DescriptorType::SIFT;
        }
    }

    inline int toOldDescriptorType(DescriptorType newValue) {
        switch (newValue) {
            case DescriptorType::SIFT: return 0;
            case DescriptorType::HoNC: return 1;
            case DescriptorType::RGBSIFT: return 2;
            case DescriptorType::vSIFT: return 3;
            case DescriptorType::DSPSIFT: return 0; // map DSPSIFT to legacy SIFT
            case DescriptorType::VGG: return 0; // no legacy mapping; treat as SIFT
            case DescriptorType::NONE: return 4;
            default: return 0;
        }
    }

    // ================================
    // STRING CONVERSION FUNCTIONS
    // ================================

    inline std::string toString(PoolingStrategy strategy) {
        switch (strategy) {
            case PoolingStrategy::NONE: return "none";
            case PoolingStrategy::DOMAIN_SIZE_POOLING: return "domain_size_pooling";
            case PoolingStrategy::STACKING: return "stacking";
            default: return "unknown";
        }
    }

    inline std::string toString(DescriptorType type) {
        switch (type) {
            case DescriptorType::SIFT: return "sift";
            case DescriptorType::HoNC: return "honc";
            case DescriptorType::RGBSIFT: return "rgbsift";
            case DescriptorType::vSIFT: return "vsift";
            case DescriptorType::DSPSIFT: return "dspsift";
            case DescriptorType::VGG: return "vgg";
            case DescriptorType::DNN_PATCH: return "dnn_patch";
            case DescriptorType::LIBTORCH_HARDNET: return "libtorch_hardnet";
            case DescriptorType::LIBTORCH_SOSNET: return "libtorch_sosnet";
            case DescriptorType::LIBTORCH_L2NET: return "libtorch_l2net";
            case DescriptorType::NONE: return "none";
            default: return "unknown";
        }
    }

    inline std::string toString(NormalizationStage stage) {
        switch (stage) {
            case NormalizationStage::BEFORE_POOLING: return "before_pooling";
            case NormalizationStage::AFTER_POOLING: return "after_pooling";
            case NormalizationStage::NO_NORMALIZATION: return "no_normalization";
            default: return "unknown";
        }
    }

    inline std::string toString(RootingStage stage) {
        switch (stage) {
            case RootingStage::R_BEFORE_POOLING: return "before_pooling";
            case RootingStage::R_AFTER_POOLING: return "after_pooling";
            case RootingStage::R_NONE: return "none";
            default: return "unknown";
        }
    }

    inline std::string toString(KeypointGenerator generator) {
        switch (generator) {
            case KeypointGenerator::SIFT: return "sift";
            case KeypointGenerator::HARRIS: return "harris";
            case KeypointGenerator::ORB: return "orb";
            case KeypointGenerator::LOCKED_IN: return "locked_in";
            default: return "unknown";
        }
    }

    inline std::string toString(KeypointSource source) {
        switch (source) {
            case KeypointSource::HOMOGRAPHY_PROJECTION: return "homography_projection";
            case KeypointSource::INDEPENDENT_DETECTION: return "independent_detection";
            default: return "unknown";
        }
    }

    inline KeypointSource keypointSourceFromString(const std::string& str) {
        if (str == "homography_projection") return KeypointSource::HOMOGRAPHY_PROJECTION;
        if (str == "independent_detection") return KeypointSource::INDEPENDENT_DETECTION;
        return KeypointSource::HOMOGRAPHY_PROJECTION; // Default to controlled evaluation
    }

    inline std::string toString(MatchingMethod method) {
        switch (method) {
            case MatchingMethod::BRUTE_FORCE: return "brute_force";
            case MatchingMethod::FLANN: return "flann";
            default: return "unknown";
        }
    }

    inline std::string toString(ValidationMethod method) {
        switch (method) {
            case ValidationMethod::HOMOGRAPHY: return "homography";
            case ValidationMethod::CROSS_IMAGE: return "cross_image";
            case ValidationMethod::NONE: return "none";
            default: return "unknown";
        }
    }

    // ================================
    // DSP SCALE WEIGHTING
    // ================================
    enum class ScaleWeighting {
        UNIFORM,
        TRIANGULAR,
        GAUSSIAN
    };

    // ================================
    // ENHANCED CONFIGURATION STRUCTURES
    // ================================

    // Parameter structures for configuration
    struct KeypointParams {
        int max_features = 2000;
        float contrast_threshold = 0.04f;
        float edge_threshold = 10.0f;
        float sigma = 1.6f;
        int num_octaves = 4;
        bool use_locked_keypoints = false;  // LEGACY: for backward compatibility
        KeypointSource source = KeypointSource::HOMOGRAPHY_PROJECTION;  // NEW: keypoint source strategy
        std::string keypoint_set_name;      // NEW: specific keypoint set to use
        std::string locked_keypoints_path;
    };

    struct DescriptorParams {
        PoolingStrategy pooling = PoolingStrategy::NONE;
        std::vector<float> scales = {1.0f, 1.5f, 2.0f};
        std::vector<float> scale_weights; // optional: if provided, use weighted pooling aligned with scales
        ScaleWeighting scale_weighting = ScaleWeighting::UNIFORM; // procedural weighting if explicit weights not provided
        float scale_weight_sigma = 0.15f; // gaussian sigma in log-space (triangular radius proxy)
        bool normalize_before_pooling = false;
        bool normalize_after_pooling = true;
        int norm_type = cv::NORM_L2;
        bool use_color = false;

        // For stacking
        DescriptorType secondary_descriptor = DescriptorType::SIFT;
        float stacking_weight = 0.5f;

        // DNN patch descriptor params (optional)
        std::string dnn_model_path;   // ONNX model path
        int dnn_input_size = 32;      // square input (e.g., 32x32)
        float dnn_support_multiplier = 1.0f; // side = multiplier * keypoint.size
        bool dnn_rotate_upright = true;      // rotate patch to keypoint orientation
        float dnn_mean = 0.0f;        // simple mean/std normalization
        float dnn_std = 1.0f;
        bool dnn_per_patch_standardize = false; // if true, standardize each patch (zero mean, unit var)
    };

    struct EvaluationParams {
        MatchingMethod matching_method = MatchingMethod::BRUTE_FORCE;
        int norm_type = cv::NORM_L2;
        bool cross_check = true;
        float match_threshold = 0.8f;

        ValidationMethod validation_method = ValidationMethod::HOMOGRAPHY;
        float validation_threshold = 0.05f; // pixels
        int min_matches_for_homography = 10;
    };

    struct DatabaseParams {
        bool enabled = false;
        std::string connection_string = "sqlite:///experiments.db";
        bool save_keypoints = true;
        bool save_descriptors = false;
        bool save_matches = false;
        bool save_visualizations = true;
    };

    // ================================
    // EXPERIMENT RESULTS STRUCTURES
    // ================================

    struct ExperimentMetrics {
        float precision = 0.0f;
        float recall = 0.0f;
        float f1_score = 0.0f;

        // Timing metrics
        double keypoint_extraction_time_ms = 0.0;
        double descriptor_extraction_time_ms = 0.0;
        double matching_time_ms = 0.0;

        // Count metrics
        int keypoints_detected = 0;
        int descriptors_computed = 0;
        int matches_found = 0;
        int correct_matches = 0;

        // Resource metrics
        double memory_peak_mb = 0.0;
    };

    struct ExperimentResults {
        std::string experiment_name;
        std::string scene_name;
        std::string descriptor_name;
        std::string keypoint_generator_name;

        ExperimentMetrics metrics;

        // Optional data
        std::vector<cv::KeyPoint> keypoints_image1;
        std::vector<cv::KeyPoint> keypoints_image2;
        cv::Mat descriptors_image1;
        cv::Mat descriptors_image2;
        std::vector<cv::DMatch> matches;

        // Paths to saved outputs
        std::string output_directory;
        std::string visualization_path;

        // Metadata
        std::string timestamp;
        std::string config_hash;
    };

} // namespace thesis_project
