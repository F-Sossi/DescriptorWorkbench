#include <gtest/gtest.h>
#include "src/core/config/YAMLConfigLoader.hpp"

using thesis_project::config::YAMLConfigLoader;

TEST(YAMLValidationErrors, MissingDatasetPathUsesDefault) {
    const char* yaml = R"YAML(
experiment: { name: t }
dataset: { type: hpatches }
descriptors: [ { name: sift, type: sift, pooling: none } ]
)YAML";
    EXPECT_NO_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); EXPECT_FALSE(cfg.dataset.path.empty()); });
}

TEST(YAMLValidationErrors, EmptyDescriptors) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors: []
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, InvalidStackingWeight) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - name: stack
    type: sift
    pooling: stacking
    stacking_weight: 1.5
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, InvalidKeypointParams) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  num_octaves: 0
descriptors: [ { name: sift, type: sift, pooling: none } ]
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, InvalidSigma) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  sigma: 0.0
descriptors: [ { name: sift, type: sift, pooling: none } ]
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, MatchingThresholdOutOfRange) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors: [ { name: sift, type: sift, pooling: none } ]
evaluation:
  matching: { method: brute_force, threshold: 1.5 }
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}
