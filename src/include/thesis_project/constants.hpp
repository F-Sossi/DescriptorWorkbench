#pragma once

namespace thesis_project {
    namespace constants {
        
        // SIFT constants (from your existing VanillaSIFT implementation)
        constexpr int SIFT_DESCR_WIDTH = 4;
        constexpr int SIFT_DESCR_HIST_BINS = 8;
        constexpr int SIFT_DESCRIPTOR_SIZE = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS; // 128
        
        // RGB SIFT constants
        constexpr int RGB_SIFT_DESCRIPTOR_SIZE = 3 * SIFT_DESCRIPTOR_SIZE; // 384
        
        // Default paths
        constexpr const char* DEFAULT_DATA_PATH = "data/hpatches";
        constexpr const char* DEFAULT_RESULTS_PATH = "results";
        constexpr const char* DEFAULT_CONFIG_PATH = "config";
        constexpr const char* DEFAULT_DATABASE_PATH = "db/experiments.db";
        
        // File extensions
        constexpr const char* YAML_EXTENSION = ".yaml";
        constexpr const char* CSV_EXTENSION = ".csv";
        constexpr const char* IMAGE_EXTENSION = ".png";
        
        // Performance thresholds
        constexpr float DEFAULT_MATCH_THRESHOLD = 0.8f;
        constexpr float DEFAULT_HOMOGRAPHY_THRESHOLD = 0.05f; // pixels
        constexpr int MIN_MATCHES_FOR_EVALUATION = 10;
        
        // Memory limits
        constexpr size_t MAX_MEMORY_MB = 4096; // 4GB limit for descriptors
        constexpr int MAX_KEYPOINTS_PER_IMAGE = 5000;
        
        // Timing constants
        constexpr int TIMING_WARMUP_ITERATIONS = 3;
        constexpr int TIMING_MEASUREMENT_ITERATIONS = 10;
        
    } // namespace constants
} // namespace thesis_project