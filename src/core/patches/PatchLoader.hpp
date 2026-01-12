#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

namespace thesis_project {
namespace patches {

/**
 * @brief Loads pre-extracted patches from HPatches dataset
 *
 * HPatches stores patches as stacked PNG files:
 * - Width: 65 pixels (patch width)
 * - Height: N × 65 pixels (patches stacked vertically)
 * - Format: 8-bit grayscale
 *
 * Each scene contains:
 * - ref.png: Reference patches
 * - e1-e5.png: Easy patches (low geometric noise)
 * - h1-h5.png: Hard patches (high geometric noise)
 * - t1-t5.png: Tough patches (additional difficulty)
 */
class PatchLoader {
public:
    static constexpr int PATCH_SIZE = 65;

    /**
     * @brief Container for a set of patches from one PNG file
     */
    struct PatchSet {
        std::vector<cv::Mat> patches;  ///< Individual 65x65 grayscale patches
        std::string name;               ///< e.g., "ref", "e1", "h3", "t2"
        std::string scene_name;         ///< Parent scene name

        size_t size() const { return patches.size(); }
        bool empty() const { return patches.empty(); }
    };

    /**
     * @brief Container for all patch sets in a scene
     */
    struct ScenePatches {
        std::string scene_name;
        PatchSet ref;                           ///< Reference patches
        std::map<std::string, PatchSet> easy;   ///< e1-e5 patches
        std::map<std::string, PatchSet> hard;   ///< h1-h5 patches
        std::map<std::string, PatchSet> tough;  ///< t1-t5 patches (if present)

        size_t numPatches() const { return ref.size(); }
    };

    /**
     * @brief Load all patches from a stacked PNG file
     * @param png_path Path to the stacked PNG file
     * @param color If true, load as color (3-channel BGR)
     * @return PatchSet containing individual 65x65 patches
     */
    static PatchSet loadStackedPNG(const std::string& png_path, bool color = false);

    /**
     * @brief Load a complete scene (ref + easy + hard + tough)
     * @param scene_dir Path to the scene directory
     * @param color If true, load patches as color
     * @return ScenePatches containing all patch sets for the scene
     */
    static ScenePatches loadScene(const std::string& scene_dir, bool color = false);

    /**
     * @brief Resize a 65x65 patch for CNN input
     * @param patch65 Input 65x65 patch
     * @param target_size Target size (default 32 for CNN)
     * @return Resized patch using INTER_AREA for anti-aliasing
     */
    static cv::Mat resizeForCNN(const cv::Mat& patch65, int target_size = 32);

    /**
     * @brief Resize a batch of patches for CNN input
     * @param patches65 Vector of 65x65 patches
     * @param target_size Target size (default 32)
     * @return Vector of resized patches
     */
    static std::vector<cv::Mat> resizeForCNN(const std::vector<cv::Mat>& patches65,
                                              int target_size = 32);

    /**
     * @brief Get the number of patches in a stacked PNG without loading all data
     * @param png_path Path to the stacked PNG file
     * @return Number of patches (height / PATCH_SIZE)
     */
    static int countPatches(const std::string& png_path);

    /**
     * @brief List all scene directories in an HPatches dataset
     * @param dataset_path Path to hpatches-release directory
     * @return Vector of scene directory paths
     */
    static std::vector<std::string> listScenes(const std::string& dataset_path);

    /**
     * @brief Check if a scene is illumination (i_*) or viewpoint (v_*)
     * @param scene_name Scene directory name
     * @return true if illumination scene (i_* prefix)
     */
    static bool isIlluminationScene(const std::string& scene_name);

    /**
     * @brief Check if a scene is viewpoint (v_*)
     * @param scene_name Scene directory name
     * @return true if viewpoint scene (v_* prefix)
     */
    static bool isViewpointScene(const std::string& scene_name);

private:
    /**
     * @brief Extract individual patches from a stacked image
     * @param stacked_img The vertically stacked patch image
     * @return Vector of individual 65x65 patches
     */
    static std::vector<cv::Mat> extractPatches(const cv::Mat& stacked_img);
};

} // namespace patches
} // namespace thesis_project
