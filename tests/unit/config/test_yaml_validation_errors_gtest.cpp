#include <gtest/gtest.h>
#include "src/core/config/YAMLConfigLoader.hpp"
#include "thesis_project/types.hpp"

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

TEST(YAMLValidationErrors, ExplicitAssignmentRequiresDescriptorKeypointSet) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  source: database
  keypoint_set_name: independent
  alternative_keypoints:
    - keypoint_set_name: paired_a
descriptors:
  - name: missing_set
    type: sift
    pooling: none
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, ExplicitAssignmentAllowsDescriptorOverride) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  source: database
  keypoint_set_name: independent
  alternative_keypoints:
    - keypoint_set_name: paired_a
descriptors:
  - name: explicit_sift
    type: sift
    pooling: none
    keypoint_set_name: paired_a
)YAML";
    EXPECT_NO_THROW({
        auto cfg = YAMLConfigLoader::loadFromString(yaml);
        EXPECT_EQ(cfg.keypoints.assignment_mode, thesis_project::KeypointAssignmentMode::EXPLICIT_ONLY);
    });
}

TEST(YAMLValidationErrors, ExplicitAssignmentRequiresCompositeComponentSets) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  source: database
  keypoint_set_name: none
  alternative_keypoints:
    - keypoint_set_name: paired_a
    - keypoint_set_name: paired_b
descriptors:
  - name: composite_missing_components
    type: composite
    aggregation: average
    components:
      - descriptor: sift
      - descriptor: sift
)YAML";
    EXPECT_THROW({ auto cfg = YAMLConfigLoader::loadFromString(yaml); (void)cfg; }, std::runtime_error);
}

TEST(YAMLValidationErrors, ExplicitAssignmentCompositeWithComponentSets) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
keypoints:
  generator: sift
  source: database
  keypoint_set_name: none
  alternative_keypoints:
    - keypoint_set_name: paired_a
    - keypoint_set_name: paired_b
descriptors:
  - name: composite_ok
    type: composite
    aggregation: weighted_avg
    components:
      - descriptor: sift
        weight: 0.5
        keypoint_set_name: paired_a
      - descriptor: sift
        weight: 0.5
        keypoint_set_name: paired_b
)YAML";
    EXPECT_NO_THROW({
        auto cfg = YAMLConfigLoader::loadFromString(yaml);
        EXPECT_EQ(cfg.keypoints.assignment_mode, thesis_project::KeypointAssignmentMode::EXPLICIT_ONLY);
    });
}
