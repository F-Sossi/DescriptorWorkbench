#ifndef THESIS_PROJECT_DATABASE_MANAGER_HPP
#define THESIS_PROJECT_DATABASE_MANAGER_HPP

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

namespace thesis_project {
namespace database {

// Forward declarations
struct ExperimentResults;
struct ExperimentConfig;
struct DatabaseConfig;

/**
 * @brief Optional database manager for experiment tracking
 *
 * This class provides experiment tracking capabilities without disrupting
 * the existing workflow. All methods are safe to call - if database is
 * disabled, they silently do nothing.
 */
class DatabaseManager {
public:
    /**
     * @brief Construct database manager
     * @param config Database configuration (connection string, enabled flag, etc.)
     */
    explicit DatabaseManager(const DatabaseConfig& config);

    /**
     * @brief Construct with simple parameters
     * @param db_path Path to SQLite database file
     * @param enabled Whether database tracking is enabled
     */
    DatabaseManager(const std::string& db_path, bool enabled = false);

    /**
     * @brief Destructor - safely closes database connection
     */
    ~DatabaseManager();

    // Copy/move operations
    DatabaseManager(const DatabaseManager&) = delete;
    DatabaseManager& operator=(const DatabaseManager&) = delete;
    DatabaseManager(DatabaseManager&&) = default;
    DatabaseManager& operator=(DatabaseManager&&) = default;

    /**
     * @brief Check if database is enabled and working
     * @return true if database operations will work
     */
    bool isEnabled() const;

    /**
     * @brief Optimize database performance for bulk operations
     * @return true if optimizations were applied successfully
     */
    bool optimizeForBulkOperations() const;

    /**
     * @brief Record experiment results
     * @param results Results from descriptor comparison experiment
     * @return true if successfully recorded (or disabled), false on error
     */
    bool recordExperiment(const ExperimentResults& results) const;

    /**
     * @brief Record experiment configuration
     * @param config Configuration used for experiment
     * @return experiment_id for linking results, or -1 if disabled/error
     */
    int recordConfiguration(const ExperimentConfig& config) const;

    /**
     * @brief Get recent experiment results
     * @param limit Maximum number of results to return
     * @return Vector of recent experiment results
     */
    std::vector<ExperimentResults> getRecentResults(int limit = 10) const;

    /**
     * @brief Get experiment statistics
     * @return Map of statistic name -> value
     */
    std::map<std::string, double> getStatistics() const;

    /**
     * @brief Store locked-in keypoints for a specific scene and image
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm")
     * @param keypoints Vector of OpenCV keypoints to store
     * @return true if successfully stored
     */
    bool storeLockedKeypoints(const std::string& scene_name, const std::string& image_name, const std::vector<cv::KeyPoint>& keypoints) const;

    /**
     * @brief Store locked-in keypoints with boundary validation
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm") 
     * @param keypoints Vector of OpenCV keypoints to store
     * @param image_width Width of the target image for boundary checking
     * @param image_height Height of the target image for boundary checking
     * @param border_buffer Buffer distance from image edges (default 0)
     * @return true if successfully stored
     */
    bool storeLockedKeypointsWithBounds(const std::string& scene_name, const std::string& image_name, 
                                       const std::vector<cv::KeyPoint>& keypoints,
                                       int image_width, int image_height, int border_buffer = 0) const;

    /**
     * @brief Retrieve locked-in keypoints for a specific scene and image
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm")
     * @return Vector of OpenCV keypoints (empty if not found or disabled)
     */
    std::vector<cv::KeyPoint> getLockedKeypoints(const std::string& scene_name, const std::string& image_name) const;

    /**
     * @brief Get all available scenes with locked keypoints
     * @return Vector of scene names
     */
    std::vector<std::string> getAvailableScenes() const;

    /**
     * @brief Get all available images for a specific scene
     * @param scene_name Name of the scene
     * @return Vector of image names
     */
    std::vector<std::string> getAvailableImages(const std::string& scene_name) const;

    /**
     * @brief Delete all locked keypoints for a specific scene
     * @param scene_name Name of the scene to clear
     * @return true if successful
     */
    bool clearSceneKeypoints(const std::string& scene_name) const;

    /**
     * @brief Create a new keypoint set with metadata
     * @param name Unique name for this keypoint set
     * @param generator_type Type of keypoint generator (e.g., "SIFT", "ORB") 
     * @param generation_method Method used ("homography_projection" or "independent_detection")
     * @param max_features Maximum features per image (0 for unlimited)
     * @param dataset_path Path to dataset used
     * @param description Human-readable description
     * @param boundary_filter_px Boundary filter applied in pixels
     * @return keypoint_set_id if successful, -1 on error
     */
    int createKeypointSet(const std::string& name,
                         const std::string& generator_type,
                         const std::string& generation_method,
                         int max_features = 2000,
                         const std::string& dataset_path = "",
                         const std::string& description = "",
                         int boundary_filter_px = 40) const;

    /**
     * @brief Create a new keypoint set with overlap filtering support
     * @param name Unique name for the keypoint set
     * @param generator_type Type of detector used (e.g., "SIFT", "Harris", "ORB")
     * @param generation_method Method used ("homography_projection", "independent_detection", "non_overlapping_detection")
     * @param max_features Maximum number of features to retain
     * @param dataset_path Path to the dataset used
     * @param description Human-readable description
     * @param boundary_filter_px Boundary filter applied in pixels
     * @param overlap_filtering Whether non-overlapping constraint was applied
     * @param min_distance Minimum distance between keypoints (0.0 if no constraint)
     * @return keypoint_set_id if successful, -1 on error
     */
    int createKeypointSetWithOverlap(const std::string& name,
                                   const std::string& generator_type,
                                   const std::string& generation_method,
                                   int max_features = 2000,
                                   const std::string& dataset_path = "",
                                   const std::string& description = "",
                                   int boundary_filter_px = 40,
                                   bool overlap_filtering = false,
                                   float min_distance = 0.0f) const;

    /**
     * @brief Store locked-in keypoints for a specific keypoint set
     * @param keypoint_set_id ID of the keypoint set
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm")
     * @param keypoints Vector of OpenCV keypoints to store
     * @return true if successfully stored
     */
    bool storeLockedKeypointsForSet(int keypoint_set_id, const std::string& scene_name, 
                                   const std::string& image_name, const std::vector<cv::KeyPoint>& keypoints) const;

    /**
     * @brief Retrieve locked-in keypoints from a specific keypoint set
     * @param keypoint_set_id ID of the keypoint set
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm")
     * @return Vector of OpenCV keypoints (empty if not found or disabled)
     */
    std::vector<cv::KeyPoint> getLockedKeypointsFromSet(int keypoint_set_id, const std::string& scene_name, 
                                                       const std::string& image_name) const;

    /**
     * @brief Get all available keypoint sets
     * @return Vector of {id, name, generation_method} tuples
     */
    std::vector<std::tuple<int, std::string, std::string>> getAvailableKeypointSets() const;

    /**
     * @brief Store descriptors for keypoints in an experiment
     * @param experiment_id ID of the experiment these descriptors belong to
     * @param scene_name Name of the scene (e.g., "i_dome", "v_wall")
     * @param image_name Name of the image (e.g., "1.ppm")
     * @param keypoints Vector of keypoints (for position linking)
     * @param descriptors cv::Mat of descriptors (rows = keypoints, cols = descriptor dimension)
     * @param processing_method String describing processing method (e.g., "SIFT-BW-None-NoNorm-NoRoot-L2")
     * @param normalization_applied Type of normalization applied
     * @param rooting_applied Type of rooting applied
     * @param pooling_applied Type of pooling applied
     * @return true if successfully stored
     */
    bool storeDescriptors(int experiment_id,
                         const std::string& scene_name,
                         const std::string& image_name,
                         const std::vector<cv::KeyPoint>& keypoints,
                         const cv::Mat& descriptors,
                         const std::string& processing_method,
                         const std::string& normalization_applied = "",
                         const std::string& rooting_applied = "",
                         const std::string& pooling_applied = "") const;

    /**
     * @brief Retrieve descriptors for a specific experiment and scene/image
     * @param experiment_id ID of the experiment
     * @param scene_name Name of the scene
     * @param image_name Name of the image
     * @return cv::Mat of descriptors (empty if not found or disabled)
     */
    cv::Mat getDescriptors(int experiment_id,
                          const std::string& scene_name,
                          const std::string& image_name) const;

    /**
     * @brief Retrieve descriptors with specific processing parameters
     * @param processing_method Processing method to filter by
     * @param normalization_applied Normalization type to filter by (optional)
     * @param rooting_applied Rooting type to filter by (optional)
     * @return Vector of {scene, image, descriptors} tuples
     */
    std::vector<std::tuple<std::string, std::string, cv::Mat>> getDescriptorsByMethod(
        const std::string& processing_method,
        const std::string& normalization_applied = "",
        const std::string& rooting_applied = "") const;

    /**
     * @brief Get all unique processing methods stored in database
     * @return Vector of processing method strings
     */
    std::vector<std::string> getAvailableProcessingMethods() const;

    /**
     * @brief Initialize database tables (safe to call multiple times)
     * @return true if successful or disabled
     */
    bool initializeTables() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Configuration for database connection
 */
struct DatabaseConfig {
    std::string connection_string;
    bool enabled = false;
    int connection_timeout = 30;
    bool create_if_missing = true;

    // Factory methods
    static DatabaseConfig disabled() {
        return DatabaseConfig{};
    }

    static DatabaseConfig sqlite(const std::string& path) {
        DatabaseConfig config;
        config.connection_string = path;
        config.enabled = true;
        return config;
    }
};

/**
 * @brief Experiment results structure for database storage
 */
struct ExperimentResults {
    int experiment_id = -1;
    std::string descriptor_type;
    std::string dataset_name;
    // Primary MAP metrics (IR-style)
    double true_map_macro = 0.0;           // Primary: Scene-balanced mAP
    double true_map_micro = 0.0;           // Overall mAP
    double true_map_macro_with_zeros = 0.0; // Conservative: includes R=0 queries
    double true_map_micro_with_zeros = 0.0; // Conservative: includes R=0 queries
    // Precision@K metrics
    double precision_at_1 = 0.0;
    double precision_at_5 = 0.0;
    double recall_at_1 = 0.0;
    double recall_at_5 = 0.0;
    // Legacy/compatibility
    double mean_average_precision = 0.0;   // Now derived from true_map_macro
    double legacy_mean_precision = 0.0;    // Original arithmetic mean for backward compatibility
    // Counts and timing
    int total_matches = 0;
    int total_keypoints = 0;
    double processing_time_ms = 0.0;
    std::string timestamp;
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Experiment configuration for database storage
 */
struct ExperimentConfig {
    std::string descriptor_type;
    std::string dataset_path;
    std::string pooling_strategy;
    double similarity_threshold = 0.7;
    int max_features = 1000;
    std::map<std::string, std::string> parameters;
    std::string timestamp;
};

} // namespace database
} // namespace thesis_project

#endif // THESIS_PROJECT_DATABASE_MANAGER_HPP
