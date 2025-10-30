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
    // Legacy enums preserved from the original experiment configuration
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
     * @brief Aggregation methods for multi-scale pooling
     * Controls how descriptors from different scales are combined
     */
    enum class PoolingAggregation {
        AVERAGE,               ///< Average pooling (default for DSPSIFT)
        MAX,                   ///< Max pooling (element-wise maximum)
        MIN,                   ///< Min pooling (element-wise minimum)
        CONCATENATE,           ///< Concatenate/stack (increases dimensionality)
        WEIGHTED_AVG           ///< Weighted average (uses scale_weights)
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
        DSPSIFT_V2,            ///< Pyramid-aware DSP wrapper with configurable aggregation (VanillaSIFT only)
        DSPRGBSIFT_V2,         ///< Pyramid-aware DSP wrapper for RGBSIFT (uses native operator())
        DSPHOWH_V2,            ///< Pyramid-aware DSP wrapper for HoWH (uses native operator())
        DSPHONC_V2,            ///< Pyramid-aware DSP wrapper for HoNC (uses native operator())
        VGG,                   ///< VGG descriptor from OpenCV xfeatures2d (non-pooled)
        DNN_PATCH,             ///< ONNX-backed patch descriptor via cv::dnn
        LIBTORCH_HARDNET,      ///< LibTorch HardNet CNN descriptor
        LIBTORCH_SOSNET,       ///< LibTorch SOSNet CNN descriptor
        LIBTORCH_L2NET,        ///< LibTorch L2-Net CNN descriptor
        ORB,                   ///< OpenCV ORB binary descriptor
        SURF,                  ///< OpenCV SURF descriptor (requires opencv_contrib)
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
        SURF,      // SURF Hessian-based blob detector
        KEYNET,    // Kornia KeyNet learned detector
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
        FLANN,
        RATIO_TEST
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

    inline std::string toString(PoolingAggregation aggregation) {
        switch (aggregation) {
            case PoolingAggregation::AVERAGE: return "average";
            case PoolingAggregation::MAX: return "max";
            case PoolingAggregation::MIN: return "min";
            case PoolingAggregation::CONCATENATE: return "concatenate";
            case PoolingAggregation::WEIGHTED_AVG: return "weighted_avg";
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
            case DescriptorType::DSPSIFT_V2: return "dspsift_v2";
            case DescriptorType::VGG: return "vgg";
            case DescriptorType::DNN_PATCH: return "dnn_patch";
            case DescriptorType::LIBTORCH_HARDNET: return "libtorch_hardnet";
            case DescriptorType::LIBTORCH_SOSNET: return "libtorch_sosnet";
            case DescriptorType::LIBTORCH_L2NET: return "libtorch_l2net";
            case DescriptorType::ORB: return "orb";
            case DescriptorType::SURF: return "surf";
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
            case KeypointGenerator::SURF: return "surf";
            case KeypointGenerator::KEYNET: return "keynet";
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
            case MatchingMethod::RATIO_TEST: return "ratio_test";
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
        int keypoint_set_id = -1;           // NEW: resolved set id for fast lookups
        std::string locked_keypoints_path;
    };

    struct DescriptorParams {
        PoolingStrategy pooling = PoolingStrategy::NONE;
        PoolingAggregation pooling_aggregation = PoolingAggregation::AVERAGE; // How to combine multi-scale descriptors
        std::vector<float> scales = {1.0f, 1.5f, 2.0f};
        std::vector<float> scale_weights; // optional: if provided, use weighted pooling aligned with scales
        ScaleWeighting scale_weighting = ScaleWeighting::UNIFORM; // procedural weighting if explicit weights not provided
        float scale_weight_sigma = 0.15f; // gaussian sigma in log-space (triangular radius proxy)

        // Normalization and rooting (RootSIFT)
        bool normalize_before_pooling = false;
        bool normalize_after_pooling = true;
        RootingStage rooting_stage = RootingStage::R_NONE; // RootSIFT transformation
        int norm_type = cv::NORM_L2;

        bool use_color = false;
        std::string device = "auto";  // "auto", "cpu", "cuda"

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

        // VGG descriptor params (optional)
        int vgg_desc_type = 100;      // 100=VGG_120, 101=VGG_80, 102=VGG_64, 103=VGG_48
        float vgg_isigma = 1.4f;      // Gaussian sigma for patch extraction
        bool vgg_img_normalize = true; // Normalize image before descriptor extraction
        bool vgg_use_scale_orientation = true; // Use keypoint scale and orientation
        float vgg_scale_factor = 6.25f; // Sampling window scale (6.25 for KAZE/SURF, 6.75 for SIFT)
        bool vgg_dsc_normalize = false; // Normalize descriptor after extraction
    };

    struct ImageRetrievalParams {
        bool enabled = false;
        std::string scorer = "total_matches";
    };

    struct KeypointVerificationParams {
        bool enabled = false;                // Enable keypoint verification task (expensive!)
        int num_distractor_scenes = 10;     // Number of distractor scenes to sample per query scene
        int num_distractors_per_scene = 20; // Number of distractor keypoints per scene
        int seed = 42;                       // Random seed for reproducibility
    };

    struct KeypointRetrievalParams {
        bool enabled = false;                // Enable keypoint retrieval task (expensive!)
        int num_distractor_scenes = 10;     // Number of distractor scenes to sample per query scene
        int num_distractors_per_scene = 20; // Number of distractor keypoints per scene
        int seed = 42;                       // Random seed for reproducibility
    };

    struct EvaluationParams {
        MatchingMethod matching_method = MatchingMethod::BRUTE_FORCE;
        int norm_type = cv::NORM_L2;
        bool cross_check = true;
        float match_threshold = 0.8f;

        ValidationMethod validation_method = ValidationMethod::HOMOGRAPHY;
        float validation_threshold = 0.05f; // pixels
        int min_matches_for_homography = 10;
        ImageRetrievalParams image_retrieval;
        KeypointVerificationParams keypoint_verification;  // Bojanic et al. (2020) verification task
        KeypointRetrievalParams keypoint_retrieval;        // Bojanic et al. (2020) retrieval task
    };

    struct DatabaseParams {
        std::string connection_string = "sqlite:///experiments.db";
        bool save_keypoints = true;
        bool save_descriptors = false;
        bool save_matches = false;
        bool save_visualizations = true;
    };

    // ================================
    // PERFORMANCE CONFIGURATION
    // ================================

    struct PerformanceParams {
        int num_threads = 0;              // 0 = auto-detect (std::thread::hardware_concurrency)
        bool parallel_scenes = true;      // Enable scene-level parallelism (OpenMP)
        bool parallel_images = false;     // Enable image-level parallelism (future enhancement)
        int batch_size = 512;             // Batch size for DNN descriptors
        bool enable_profiling = false;    // Print detailed timing breakdown
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
