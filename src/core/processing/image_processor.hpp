#ifndef DESCRIPTOR_COMPARE_IMAGE_PROCESSOR_HPP
#define DESCRIPTOR_COMPARE_IMAGE_PROCESSOR_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <map>


#include "../config/experiment_config.hpp"
#include "processor_utils.hpp"
#include "src/core/metrics/ExperimentMetrics.hpp"
#include "src/core/visualization/VisualVerification.hpp"


/**
 * @brief The image_processor class provides functionality for processing image directories
 *        and performing various image processing tasks.
 */
class image_processor {
public:
    /**
     * @brief Processes a directory of image folders and performs image processing based on the provided configuration.
     * @param data_folder The path to the directory containing image folders.
     * @param results_folder The path to the directory where the results will be stored.
     * @param config The experiment configuration specifying the processing options.
     * @return ExperimentMetrics containing all computed metrics and success status.
     */
    static ExperimentMetrics process_directory(const std::string& data_folder, const std::string& results_folder, const experiment_config& config);


private:
    /**
     * @brief Processes a single image folder and performs image processing based on the provided configuration.
     * @param folder The path to the image folder.
     * @param results_folder The path to the directory where the results will be stored.
     * @param config The experiment configuration specifying the processing options.
     * @return ExperimentMetrics for this specific folder/scene.
     */
    static ExperimentMetrics process_image_folder_keypoints_unlocked(const std::string& folder, const std::string& results_folder, const experiment_config& config);
    static ExperimentMetrics process_image_folder_keypoints_locked(const std::string& folder, const std::string& results_folder, const experiment_config& config);
};


#endif //DESCRIPTOR_COMPARE_IMAGE_PROCESSOR_HPP
