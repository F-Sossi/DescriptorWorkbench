#include "KeypointGeneratorFactory.hpp"
#include "detectors/SIFTKeypointGenerator.hpp"
#include "detectors/HarrisKeypointGenerator.hpp"
#include "detectors/ORBKeypointGenerator.hpp"
#include "detectors/SURFKeypointGenerator.hpp"
#include "detectors/NonOverlappingKeypointGenerator.hpp"
#include "generators/KeynetDetector.hpp"
#include <stdexcept>
#include <algorithm>

namespace thesis_project {

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::create(
    KeypointGenerator type,
    bool non_overlapping,
    float min_distance
) {
    std::unique_ptr<IKeypointGenerator> detector;
    
    switch (type) {
        case KeypointGenerator::SIFT:
            detector = createSIFT();
            break;

        case KeypointGenerator::HARRIS:
            detector = createHarris();
            break;

        case KeypointGenerator::ORB:
            detector = createORB();
            break;

        case KeypointGenerator::SURF:
            detector = createSURF();
            break;

        case KeypointGenerator::KEYNET:
            detector = createKeyNet();
            break;

        case KeypointGenerator::LOCKED_IN:
            throw std::invalid_argument("LOCKED_IN detector type should be handled separately");

        default:
            throw std::invalid_argument("Unsupported detector type: " + std::to_string(static_cast<int>(type)));
    }
    
    // Apply non-overlapping constraint if requested
    if (non_overlapping) {
        detector = makeNonOverlapping(std::move(detector), min_distance);
    }
    
    return detector;
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createFromString(
    const std::string& detector_name,
    bool non_overlapping,
    float min_distance
) {
    KeypointGenerator type = parseDetectorType(detector_name);
    return create(type, non_overlapping, min_distance);
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createSIFT() {
    return std::make_unique<SIFTKeypointGenerator>();
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createHarris() {
    return std::make_unique<HarrisKeypointGenerator>();
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createORB() {
    return std::make_unique<ORBKeypointGenerator>();
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createSURF() {
    return std::make_unique<SURFKeypointGenerator>();
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::createKeyNet() {
    return std::make_unique<KeynetDetector>();
}

std::unique_ptr<IKeypointGenerator> KeypointGeneratorFactory::makeNonOverlapping(
    std::unique_ptr<IKeypointGenerator> base_detector,
    float min_distance
) {
    if (!base_detector) {
        throw std::invalid_argument("Base detector cannot be null");
    }
    
    return std::make_unique<NonOverlappingKeypointGenerator>(
        std::move(base_detector),
        min_distance
    );
}

KeypointGenerator KeypointGeneratorFactory::parseDetectorType(const std::string& detector_str) {
    std::string lower_str = detector_str;
    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);

    if (lower_str == "sift") {
        return KeypointGenerator::SIFT;
    } else if (lower_str == "harris") {
        return KeypointGenerator::HARRIS;
    } else if (lower_str == "orb") {
        return KeypointGenerator::ORB;
    } else if (lower_str == "surf") {
        return KeypointGenerator::SURF;
    } else if (lower_str == "keynet") {
        return KeypointGenerator::KEYNET;
    } else if (lower_str == "locked_in") {
        return KeypointGenerator::LOCKED_IN;
    } else {
        throw std::invalid_argument("Unknown detector type: " + detector_str);
    }
}

std::vector<std::string> KeypointGeneratorFactory::getSupportedDetectors() {
    return {"sift", "harris", "orb", "surf", "keynet"};
}

bool KeypointGeneratorFactory::isSupported(KeypointGenerator type) {
    switch (type) {
        case KeypointGenerator::SIFT:
        case KeypointGenerator::HARRIS:
        case KeypointGenerator::ORB:
        case KeypointGenerator::SURF:
        case KeypointGenerator::KEYNET:
            return true;
        case KeypointGenerator::LOCKED_IN:
            return false; // Handled separately
        default:
            return false;
    }
}

float KeypointGeneratorFactory::getRecommendedMinDistance(
    KeypointGenerator type,
    int descriptor_patch_size
) {
    switch (type) {
        case KeypointGenerator::SIFT:
            return static_cast<float>(descriptor_patch_size);

        case KeypointGenerator::HARRIS:
            // Harris corners are typically more sparse
            return static_cast<float>(descriptor_patch_size * 0.8f);

        case KeypointGenerator::ORB:
            // ORB uses 31x31 patches by default
            return static_cast<float>(std::max(descriptor_patch_size, 31));

        case KeypointGenerator::SURF:
            // SURF typically uses ~20px patches
            return static_cast<float>(descriptor_patch_size);

        case KeypointGenerator::KEYNET:
            // KeyNet is designed for CNN descriptors with typical patch sizes
            return static_cast<float>(descriptor_patch_size);

        default:
            return static_cast<float>(descriptor_patch_size);
    }
}

} // namespace thesis_project