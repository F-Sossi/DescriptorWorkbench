#ifndef DESCRIPTOR_COMPARE_EXPERIMENT_CONFIG_HPP
#define DESCRIPTOR_COMPARE_EXPERIMENT_CONFIG_HPP

#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> //x
#include <vector>
#include <memory>
#include <string>

/**
 * @brief Enum for different pooling strategies
 */
enum PoolingStrategy {
    NONE, ///< No pooling
    DOMAIN_SIZE_POOLING, ///< Domain size pooling
    STACKING ///< Stacking pooling
};

/**
 * @brief Enum for when to apply normalization
 */
enum NormalizationStage {
    BEFORE_POOLING, ///< Normalization before pooling
    AFTER_POOLING, ///< Normalization after pooling
    NO_NORMALIZATION ///< Skip normalization
};

/**
 * @brief Enum for rooting stage
 */
enum RootingStage {
    R_BEFORE_POOLING, ///< Rooting before pooling
    R_AFTER_POOLING, ///< Rooting after pooling
    R_NONE ///< No rooting
};

/**
 * @brief Enum for matching strategies
 */
enum MatchingStrategy {
    BRUTE_FORCE, ///< Brute-force matching with cross-check
    FLANN,       ///< FLANN-based approximate matching (future)
    RATIO_TEST   ///< Lowe's ratio test (future)
};

/**
 * @brief Enum for descriptor types that can be used in the experiments
 * they should be added here and in the create DescriptorExtractor method
 * this will allow you to create the pointer with whatever options needed
 * if the descriptor inherits from Vanilla SIFT or properly implements the
 * Features2D interface they should run as expected
 */
enum DescriptorType {
    DESCRIPTOR_SIFT, ///< SIFT descriptor
    DESCRIPTOR_HoNC, ///< HoNC descriptor
    DESCRIPTOR_RGBSIFT, ///< RGBSIFT descriptor
    DESCRIPTOR_vSIFT, ///< Vanilla SIFT descriptor
    NO_DESCRIPTOR ///< No descriptor
    // Add more descriptor types as needed
};

/**
 * @brief Enum for descriptor color space
 */
enum DescriptorColorSpace {
    D_COLOR, ///< Color descriptor
    D_BW ///< Black and white descriptor
};

/**
 * @brief Enum for image type
 */
enum ImageType {
    COLOR, ///< Color image
    BW ///< Black and white image
};

/**
 * @brief Enum for verification type (only works if multithreading is disabled)
 */
enum VerificationType {
    MATCHES, ///< Verification using descriptor matches
    HOMOGRAPHY, ///< Verification using homography projection
    NO_VISUAL_VERIFICATION ///< No visual verification
};

/**
 * @brief Struct for descriptor options
 */
struct DescriptorOptions {
    PoolingStrategy poolingStrategy = NONE; ///< Pooling strategy
    NormalizationStage normalizationStage = NO_NORMALIZATION; ///< Normalization stage
    RootingStage rootingStage = R_NONE; ///< Rooting stage
    ImageType imageType = COLOR; ///< Image type
    DescriptorType descriptorType = DESCRIPTOR_SIFT; ///< Descriptor type
    DescriptorType descriptorType2 = NO_DESCRIPTOR; ///< Descriptor type
    DescriptorColorSpace descriptorColorSpace = D_BW; ///< Descriptor color space
    DescriptorColorSpace descriptorColorSpace2 = D_BW; ///< Descriptor color space
    int normType = cv::NORM_L1; ///< Normalization type
    std::vector<float> scales = {1.0f, 1.5f, 2.0f}; ///< Scales used for domain size pooling
    std::vector<float> scale_weights; ///< Optional weights aligned with scales for weighted DSP pooling
    int scale_weighting_mode = 0; ///< 0=uniform, 1=triangular, 2=gaussian
    float scale_weight_sigma = 0.15f; ///< Sigma for gaussian weighting (log-space) or radius proxy for triangular
    bool recordKeypoints = false; ///< Flag to enable/disable recording keypoints
    bool recordDescriptors = false; ///< Flag to enable/disable recording descriptors
    bool UseLockedInKeypoints = true; ///< Flag to enable/disable locked-in keypoints
    int max_features = 0; ///< Maximum number of keypoints to detect (0 = unlimited)

    DescriptorOptions() = default; ///< Default constructor
};

/**
 * @brief Struct for experiment configuration
 */
class experiment_config {
public:

    VerificationType verificationType = NO_VISUAL_VERIFICATION;
    experiment_config(); ///< Default constructor
    DescriptorOptions descriptorOptions; ///< Descriptor options
    //DescriptorType descriptorType; ///< Descriptor type
    cv::Ptr<cv::Feature2D> detector; ///< Descriptor extractor
    cv::Ptr<cv::Feature2D> detector2; ///< Descriptor extractor
    bool useMultiThreading = true; ///< Flag to enable/disable multithreading
    double matchThreshold = 0.05; ///< Match threshold
    MatchingStrategy matchingStrategy = BRUTE_FORCE; ///< Matching strategy
    int experiment_id = -1; ///< Database experiment ID for descriptor storage

    /**
     * @brief Constructor with descriptor options and type
     * @param options Descriptor options
     * @param type Descriptor type
     */
    [[maybe_unused]] explicit experiment_config(const DescriptorOptions& options);

    /**
     * @brief Constructor with configuration file path
     * @param configFilePath Configuration file path
     */
    [[maybe_unused]] explicit experiment_config(const std::string& configFilePath);

    /**
     * @brief Loads experiment configuration from a file
     * @param configFilePath Configuration file path
     */
    void loadFromFile(const std::string& configFilePath);

    /**
     * @brief Sets the descriptor options
     * @param options Descriptor options
     */
    void setDescriptorOptions(const DescriptorOptions& options);

    /**
     * @brief Sets the descriptor type
     * @param type Descriptor type
     */
    void setDescriptorType(DescriptorType type);

    /**
     * @brief Sets the pooling strategy
     * @param strategy Pooling strategy
     */
    void setPoolingStrategy(PoolingStrategy strategy);

    /**
     * @brief Sets the scales for domain size pooling
     * @param scales Scales
     */
    void setScales(const std::vector<float>& scales);

    /**
     * @brief Sets the normalization type
     * @param normType Normalization type
     */
    void setNormType(int normType);

    /**
     * @brief Sets the normalization stage
     * @param stage Normalization stage
     */
    void setNormalizationStage(NormalizationStage stage);

    /**
     * @brief Sets the rooting stage
     * @param stage Rooting stage
     */
    void setRootingStage(RootingStage stage);

    /**
     * @brief Sets the image type
     * @param type Image type
     */
    void setImageType(ImageType type);

    /**
     * @brief Sets the descriptor color space
     * @param colorSpace Descriptor color space
     */
    void setDescriptorColorSpace(DescriptorColorSpace colorSpace);

    void setVerificationType(VerificationType inputVerificationType);

    /**
     * @brief Sets the flag to enable/disable multithreading
     * @param useMultiThreading Flag to enable/disable multithreading
     */
    void setUseMultiThreading(bool useMultiThreading);

    /**
     * @brief Creates the descriptor extractor based on the descriptor type
     * @return Descriptor extractor
     */
    static cv::Ptr<cv::Feature2D> createDescriptorExtractor(DescriptorType type, int max_features = 0);
    
    // Method to refresh detectors after configuration changes
    void refreshDetectors();

    void verifyConfiguration();

private:


};

#endif // DESCRIPTOR_COMPARE_EXPERIMENT_CONFIG_HPP
