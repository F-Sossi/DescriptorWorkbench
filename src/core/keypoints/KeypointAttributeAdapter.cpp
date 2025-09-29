#include "thesis_project/keypoints/KeypointAttributeAdapter.hpp"
#include "thesis_project/types.hpp"
#include <unordered_map>

namespace thesis_project::keypoints {

namespace {

    inline std::string detectorKey(KeypointGenerator detector) {
        return toString(detector);
    }

} // namespace

std::optional<KeypointGenerator> preferredDetectorForDescriptor(DescriptorType descriptor_type) {
    switch (descriptor_type) {
        case DescriptorType::SIFT:
        case DescriptorType::RGBSIFT:
        case DescriptorType::HoNC:
        case DescriptorType::vSIFT:
        case DescriptorType::DSPSIFT:
        case DescriptorType::SURF:
        case DescriptorType::VGG:
            return KeypointGenerator::SIFT;

        case DescriptorType::ORB:
            return KeypointGenerator::ORB;

        case DescriptorType::DNN_PATCH:
        case DescriptorType::LIBTORCH_HARDNET:
        case DescriptorType::LIBTORCH_SOSNET:
        case DescriptorType::LIBTORCH_L2NET:
            return KeypointGenerator::KEYNET;

        default:
            return std::nullopt;
    }
}

HydrationStats loadKeypointsWithAttributes(
    database::DatabaseManager& db,
    int keypoint_set_id,
    const std::string& scene_name,
    const std::string& image_name,
    KeypointGenerator detector,
    std::vector<cv::KeyPoint>& out_keypoints) {

    HydrationStats stats;
    out_keypoints.clear();

    auto records = db.getLockedKeypointsWithIds(keypoint_set_id, scene_name, image_name);
    if (records.empty()) {
        return stats;
    }

    stats.total = records.size();

    auto attribute_map = db.getDetectorAttributesForImage(
        keypoint_set_id,
        scene_name,
        image_name,
        detectorKey(detector));

    out_keypoints.reserve(records.size());

    for (const auto& record : records) {
        auto kp = record.keypoint;
        auto attr_it = attribute_map.find(record.id);
        if (attr_it != attribute_map.end()) {
            const auto& attr = attr_it->second;
            kp.size = attr.size;
            kp.angle = attr.angle;
            kp.response = attr.response;
            kp.octave = attr.octave;
            kp.class_id = attr.class_id;
            ++stats.hydrated;
        }
        out_keypoints.push_back(kp);
    }

    return stats;
}

} // namespace thesis_project::keypoints

