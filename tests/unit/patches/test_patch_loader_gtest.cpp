#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

#include "src/core/patches/PatchLoader.hpp"

using namespace thesis_project::patches;

// Test fixture for PatchLoader tests
class PatchLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for test files
        temp_dir_ = std::filesystem::temp_directory_path() / "patch_loader_test";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        // Clean up temporary files
        std::filesystem::remove_all(temp_dir_);
    }

    // Helper to create a stacked patch PNG (multiple 65x65 patches vertically)
    std::string createStackedPNG(const std::string& name, int num_patches, bool color = false) {
        int height = num_patches * PatchLoader::PATCH_SIZE;
        int width = PatchLoader::PATCH_SIZE;

        cv::Mat img;
        if (color) {
            img = cv::Mat(height, width, CV_8UC3);
            cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        } else {
            img = cv::Mat(height, width, CV_8UC1);
            cv::randu(img, 0, 255);
        }

        std::string path = (temp_dir_ / (name + ".png")).string();
        cv::imwrite(path, img);
        return path;
    }

    // Helper to create a mock scene directory
    std::string createMockScene(const std::string& scene_name, int patches_per_file) {
        std::filesystem::path scene_dir = temp_dir_ / scene_name;
        std::filesystem::create_directories(scene_dir);

        // Create ref.png
        cv::Mat ref(patches_per_file * PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
        cv::randu(ref, 0, 255);
        cv::imwrite((scene_dir / "ref.png").string(), ref);

        // Create e1-e5 (easy)
        for (int i = 1; i <= 5; ++i) {
            cv::Mat easy(patches_per_file * PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
            cv::randu(easy, 0, 255);
            cv::imwrite((scene_dir / ("e" + std::to_string(i) + ".png")).string(), easy);
        }

        // Create h1-h5 (hard)
        for (int i = 1; i <= 5; ++i) {
            cv::Mat hard(patches_per_file * PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
            cv::randu(hard, 0, 255);
            cv::imwrite((scene_dir / ("h" + std::to_string(i) + ".png")).string(), hard);
        }

        return scene_dir.string();
    }

    std::filesystem::path temp_dir_;
};

// ============================================================================
// Basic PNG Loading Tests
// ============================================================================

TEST_F(PatchLoaderTest, LoadsStackedPNGGrayscale) {
    const int num_patches = 10;
    std::string path = createStackedPNG("test_gray", num_patches, false);

    auto patch_set = PatchLoader::loadStackedPNG(path, false);

    EXPECT_EQ(patch_set.patches.size(), num_patches);
    EXPECT_EQ(patch_set.name, "test_gray");

    // Each patch should be 65x65 grayscale
    for (const auto& patch : patch_set.patches) {
        EXPECT_EQ(patch.rows, PatchLoader::PATCH_SIZE);
        EXPECT_EQ(patch.cols, PatchLoader::PATCH_SIZE);
        EXPECT_EQ(patch.channels(), 1);
    }
}

TEST_F(PatchLoaderTest, LoadsStackedPNGColor) {
    const int num_patches = 5;
    std::string path = createStackedPNG("test_color", num_patches, true);

    auto patch_set = PatchLoader::loadStackedPNG(path, true);

    EXPECT_EQ(patch_set.patches.size(), num_patches);

    // Each patch should be 65x65 color (3 channels)
    for (const auto& patch : patch_set.patches) {
        EXPECT_EQ(patch.rows, PatchLoader::PATCH_SIZE);
        EXPECT_EQ(patch.cols, PatchLoader::PATCH_SIZE);
        EXPECT_EQ(patch.channels(), 3);
    }
}

TEST_F(PatchLoaderTest, ThrowsOnMissingFile) {
    EXPECT_THROW(
        PatchLoader::loadStackedPNG("/nonexistent/path/file.png", false),
        std::runtime_error
    );
}

TEST_F(PatchLoaderTest, ThrowsOnInvalidWidth) {
    // Create an image with wrong width
    cv::Mat bad_img(130, 100, CV_8UC1);  // 100 width instead of 65
    cv::randu(bad_img, 0, 255);
    std::string path = (temp_dir_ / "bad_width.png").string();
    cv::imwrite(path, bad_img);

    EXPECT_THROW(
        PatchLoader::loadStackedPNG(path, false),
        std::runtime_error
    );
}

TEST_F(PatchLoaderTest, ThrowsOnInvalidHeight) {
    // Create an image with height not divisible by PATCH_SIZE
    cv::Mat bad_img(100, PatchLoader::PATCH_SIZE, CV_8UC1);  // 100 not divisible by 65
    cv::randu(bad_img, 0, 255);
    std::string path = (temp_dir_ / "bad_height.png").string();
    cv::imwrite(path, bad_img);

    EXPECT_THROW(
        PatchLoader::loadStackedPNG(path, false),
        std::runtime_error
    );
}

// ============================================================================
// Patch Counting Tests
// ============================================================================

TEST_F(PatchLoaderTest, CountPatchesReturnsCorrectCount) {
    const int num_patches = 15;
    std::string path = createStackedPNG("count_test", num_patches, false);

    int count = PatchLoader::countPatches(path);

    EXPECT_EQ(count, num_patches);
}

TEST_F(PatchLoaderTest, CountPatchesReturnsZeroForMissingFile) {
    int count = PatchLoader::countPatches("/nonexistent/path/file.png");
    EXPECT_EQ(count, 0);
}

// ============================================================================
// Scene Loading Tests
// ============================================================================

TEST_F(PatchLoaderTest, LoadsSceneWithAllFiles) {
    const int patches_per_file = 3;
    std::string scene_dir = createMockScene("v_test", patches_per_file);

    auto scene = PatchLoader::loadScene(scene_dir, false);

    EXPECT_EQ(scene.scene_name, "v_test");
    EXPECT_EQ(scene.ref.patches.size(), patches_per_file);
    EXPECT_EQ(scene.easy.size(), 5);  // e1-e5
    EXPECT_EQ(scene.hard.size(), 5);  // h1-h5

    // Check each easy and hard set
    for (const auto& [key, pset] : scene.easy) {
        EXPECT_EQ(pset.patches.size(), patches_per_file);
    }
    for (const auto& [key, pset] : scene.hard) {
        EXPECT_EQ(pset.patches.size(), patches_per_file);
    }
}

TEST_F(PatchLoaderTest, ThrowsOnMissingRefFile) {
    std::filesystem::path scene_dir = temp_dir_ / "bad_scene";
    std::filesystem::create_directories(scene_dir);
    // Don't create ref.png

    EXPECT_THROW(
        PatchLoader::loadScene(scene_dir.string(), false),
        std::runtime_error
    );
}

// ============================================================================
// Scene Listing Tests
// ============================================================================

TEST_F(PatchLoaderTest, ListsScenesCorrectly) {
    // Create some scene directories
    std::filesystem::create_directories(temp_dir_ / "i_scene1");
    std::filesystem::create_directories(temp_dir_ / "v_scene2");
    std::filesystem::create_directories(temp_dir_ / "i_scene3");
    std::filesystem::create_directories(temp_dir_ / "other_dir");  // Should be ignored

    auto scenes = PatchLoader::listScenes(temp_dir_.string());

    EXPECT_EQ(scenes.size(), 3);
    // Should be sorted
    EXPECT_TRUE(scenes[0].find("i_scene1") != std::string::npos);
    EXPECT_TRUE(scenes[1].find("i_scene3") != std::string::npos);
    EXPECT_TRUE(scenes[2].find("v_scene2") != std::string::npos);
}

TEST_F(PatchLoaderTest, ThrowsOnNonexistentDatasetPath) {
    EXPECT_THROW(
        PatchLoader::listScenes("/nonexistent/path"),
        std::runtime_error
    );
}

// ============================================================================
// Scene Type Detection Tests
// ============================================================================

TEST_F(PatchLoaderTest, DetectsIlluminationScene) {
    EXPECT_TRUE(PatchLoader::isIlluminationScene("i_scene1"));
    EXPECT_TRUE(PatchLoader::isIlluminationScene("/path/to/i_scene2"));
    EXPECT_FALSE(PatchLoader::isIlluminationScene("v_scene1"));
    EXPECT_FALSE(PatchLoader::isIlluminationScene("other"));
}

TEST_F(PatchLoaderTest, DetectsViewpointScene) {
    EXPECT_TRUE(PatchLoader::isViewpointScene("v_scene1"));
    EXPECT_TRUE(PatchLoader::isViewpointScene("/path/to/v_scene2"));
    EXPECT_FALSE(PatchLoader::isViewpointScene("i_scene1"));
    EXPECT_FALSE(PatchLoader::isViewpointScene("other"));
}

// ============================================================================
// CNN Resize Tests
// ============================================================================

TEST_F(PatchLoaderTest, ResizesForCNN) {
    cv::Mat patch65(PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
    cv::randu(patch65, 0, 255);

    cv::Mat patch32 = PatchLoader::resizeForCNN(patch65, 32);

    EXPECT_EQ(patch32.rows, 32);
    EXPECT_EQ(patch32.cols, 32);
    EXPECT_EQ(patch32.channels(), 1);
}

TEST_F(PatchLoaderTest, ResizesBatchForCNN) {
    std::vector<cv::Mat> patches65;
    for (int i = 0; i < 5; ++i) {
        cv::Mat patch(PatchLoader::PATCH_SIZE, PatchLoader::PATCH_SIZE, CV_8UC1);
        cv::randu(patch, 0, 255);
        patches65.push_back(patch);
    }

    auto patches32 = PatchLoader::resizeForCNN(patches65, 32);

    EXPECT_EQ(patches32.size(), 5);
    for (const auto& patch : patches32) {
        EXPECT_EQ(patch.rows, 32);
        EXPECT_EQ(patch.cols, 32);
    }
}

TEST_F(PatchLoaderTest, ResizeHandlesEmptyPatch) {
    cv::Mat empty;
    cv::Mat result = PatchLoader::resizeForCNN(empty, 32);
    EXPECT_TRUE(result.empty());
}

// ============================================================================
// Patch Data Independence Tests
// ============================================================================

TEST_F(PatchLoaderTest, ExtractedPatchesOwnTheirData) {
    const int num_patches = 3;
    std::string path = createStackedPNG("data_test", num_patches, false);

    auto patch_set = PatchLoader::loadStackedPNG(path, false);

    // Modify the first patch - should not affect others
    patch_set.patches[0].setTo(0);

    // Check that other patches are unaffected (they should have non-zero values)
    double min_val, max_val;
    cv::minMaxLoc(patch_set.patches[1], &min_val, &max_val);
    EXPECT_GT(max_val, 0) << "Patches should own their data independently";
}
