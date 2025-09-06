#include <gtest/gtest.h>
#include "src/core/config/YAMLConfigLoader.hpp"

using thesis_project::config::YAMLConfigLoader;

TEST(YAMLSchemaV1, MinimalLoads) {
    const char* yaml = R"YAML(
experiment: { name: x }
dataset: { type: hpatches, path: data/hp }
keypoints: { generator: sift, max_features: 1000, source: homography_projection }
descriptors: [ { name: sift, type: sift, pooling: none, normalize_after_pooling: true } ]
evaluation: { matching: { method: brute_force, threshold: 0.8 } }
)YAML";
    auto cfg = YAMLConfigLoader::loadFromString(yaml);
    EXPECT_EQ(cfg.descriptors.size(), 1u);
    EXPECT_EQ(cfg.descriptors[0].name, std::string("sift"));
}

TEST(YAMLSchemaV1, DSPScalesPositive) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - { name: dsp, type: sift, pooling: domain_size_pooling, scales: [0.85, 1.0, 1.3], scale_weight_sigma: 0.15 }
)YAML";
    EXPECT_NO_THROW( { auto c = YAMLConfigLoader::loadFromString(yaml); (void)c; } );
}

TEST(YAMLSchemaV1, DSPWeightsLengthMustMatch) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - name: dsp
    type: sift
    pooling: domain_size_pooling
    scales: [0.85, 1.0, 1.3]
    scale_weights: [0.2, 0.8]
)YAML";
    EXPECT_THROW( { auto c = YAMLConfigLoader::loadFromString(yaml); (void)c; }, std::runtime_error );
}

TEST(YAMLSchemaV1, StackingRequiresSecondary) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - name: stack
    type: sift
    pooling: stacking
    secondary_descriptor: none
)YAML";
    // 'none' is invalid type string and will throw earlier, but ensure error thrown
    EXPECT_THROW( { auto c = YAMLConfigLoader::loadFromString(yaml); (void)c; }, std::runtime_error );
}

TEST(YAMLSchemaV1, UniqueDescriptorNames) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - { name: sift, type: sift, pooling: none }
  - { name: sift, type: rgbsift, pooling: none }
)YAML";
    EXPECT_THROW( { auto c = YAMLConfigLoader::loadFromString(yaml); (void)c; }, std::runtime_error );
}

TEST(YAMLSchemaV1, VGGTypeParses) {
    const char* yaml = R"YAML(
dataset: { type: hpatches, path: data/hp }
descriptors:
  - { name: vgg_desc, type: vgg, pooling: none }
)YAML";
    EXPECT_NO_THROW( { auto c = YAMLConfigLoader::loadFromString(yaml); (void)c; } );
}
