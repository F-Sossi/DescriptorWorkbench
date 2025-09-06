#include "experiment_config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../keypoints/VanillaSIFT.h"
#include "../keypoints/HoNC.h"
#include "../keypoints/RGBSIFT.h"


experiment_config::experiment_config() {
    this->detector = createDescriptorExtractor(this->descriptorOptions.descriptorType, this->descriptorOptions.max_features);

    if(this->descriptorOptions.poolingStrategy == STACKING){
        this->detector2 = createDescriptorExtractor(this->descriptorOptions.descriptorType2, this->descriptorOptions.max_features);
        if(this->detector2) {
        } else {
            std::cerr << "[ERROR] Secondary detector creation failed!" << std::endl;
        }
    }

    verifyConfiguration();

}

[[maybe_unused]] experiment_config::experiment_config(const DescriptorOptions& options){

    this->detector = createDescriptorExtractor(this->descriptorOptions.descriptorType, this->descriptorOptions.max_features);

    if(this->descriptorOptions.poolingStrategy == STACKING){
        this->detector2 = createDescriptorExtractor(this->descriptorOptions.descriptorType2, this->descriptorOptions.max_features);
    }

    verifyConfiguration();
}

[[maybe_unused]] experiment_config::experiment_config(const std::string& configFilePath) {
    loadFromFile(configFilePath);
    // create descriptor pointer
    this->detector = createDescriptorExtractor(this->descriptorOptions.descriptorType, this->descriptorOptions.max_features);

    if(this->descriptorOptions.poolingStrategy == STACKING){
        this->detector2 = createDescriptorExtractor(this->descriptorOptions.descriptorType2, this->descriptorOptions.max_features);
    }

    verifyConfiguration();
}

void experiment_config::loadFromFile(const std::string& configFilePath) {
    std::ifstream file(configFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + configFilePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;  // Skip empty lines and comments
        }

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            // Trim leading and trailing whitespace from key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (key == "poolingStrategy") {
                if (value == "NONE") {
                    setPoolingStrategy(NONE);
                } else if (value == "DOMAIN_SIZE_POOLING") {
                    setPoolingStrategy(DOMAIN_SIZE_POOLING);
                } else if (value == "STACKING") {
                    setPoolingStrategy(STACKING);
                }
            } else if (key == "scales") {
                std::vector<float> scales;
                std::istringstream scaleStream(value);
                std::string scale;
                while (std::getline(scaleStream, scale, ',')) {
                    scales.push_back(std::stof(scale));
                }
                setScales(scales);
            } else if (key == "normType") {
                setNormType(std::stoi(value));
            } else if (key == "normalizationStage") {
                if (value == "BEFORE_POOLING") {
                    setNormalizationStage(BEFORE_POOLING);
                } else if (value == "AFTER_POOLING") {
                    setNormalizationStage(AFTER_POOLING);
                } else if (value == "NO_NORMALIZATION") {
                    setNormalizationStage(NO_NORMALIZATION);
                }
            } else if (key == "rootingStage") {
                if (value == "R_BEFORE_POOLING") {
                    setRootingStage(R_BEFORE_POOLING);
                } else if (value == "R_AFTER_POOLING") {
                    setRootingStage(R_AFTER_POOLING);
                } else if (value == "R_NONE") {
                    setRootingStage(R_NONE);
                }
            } else if (key == "imageType") {
                if (value == "COLOR") {
                    setImageType(COLOR);
                } else if (value == "BW") {
                    setImageType(BW);
                }
            } else if (key == "descriptorType") {
                if (value == "DESCRIPTOR_SIFT") {
                    setDescriptorType(DESCRIPTOR_SIFT);
                } else if (value == "DESCRIPTOR_HoNC") {
                    setDescriptorType(DESCRIPTOR_HoNC);
                } else if (value == "DESCRIPTOR_RGBSIFT") {
                    setDescriptorType(DESCRIPTOR_RGBSIFT);
                } else if (value == "DESCRIPTOR_vSIFT") {
                    setDescriptorType(DESCRIPTOR_vSIFT);
                }
            } else if (key == "descriptorColorSpace") {
                if (value == "D_COLOR") {
                    setDescriptorColorSpace(D_COLOR);
                } else if (value == "D_BW") {
                    setDescriptorColorSpace(D_BW);
                }
            } else if (key == "useMultiThreading") {
                setUseMultiThreading(value == "true");
            }
        }
    }

    file.close();
}

void experiment_config::setDescriptorOptions(const DescriptorOptions& options) {
    descriptorOptions = options;
}

void experiment_config::setDescriptorType(DescriptorType type) {
    this->detector = createDescriptorExtractor(type, this->descriptorOptions.max_features);
}

void experiment_config::setPoolingStrategy(PoolingStrategy strategy) {
    descriptorOptions.poolingStrategy = strategy;
}

void experiment_config::setScales(const std::vector<float>& scales) {
    descriptorOptions.scales = scales;
}

void experiment_config::setNormType(int normType) {
    descriptorOptions.normType = normType;
}

void experiment_config::setNormalizationStage(NormalizationStage stage) {
    descriptorOptions.normalizationStage = stage;
}

void experiment_config::setRootingStage(RootingStage stage) {
    descriptorOptions.rootingStage = stage;
}

void experiment_config::setImageType(ImageType type) {
    descriptorOptions.imageType = type;
}

void experiment_config::setDescriptorColorSpace(DescriptorColorSpace colorSpace) {
    descriptorOptions.descriptorColorSpace = colorSpace;
}

void experiment_config::setVerificationType(VerificationType inputVerificationType) {
    experiment_config::verificationType = inputVerificationType;
}

void experiment_config::setUseMultiThreading(bool selection) {
    this->useMultiThreading = selection;
}

/**
 * @brief Create a descriptor extractor based on the descriptor type this is where
 * and new descriptor type should be added in order to be used in the experiments
 * @return Descriptor extractor
 */
cv::Ptr<cv::Feature2D> experiment_config::createDescriptorExtractor(DescriptorType type, int max_features) {
    switch (type) {
        case DESCRIPTOR_SIFT:
            if (max_features > 0) {
                return cv::SIFT::create(max_features);
            } else {
                return cv::SIFT::create(); // Unlimited if max_features = 0
            }
        case DESCRIPTOR_vSIFT:
            return VanillaSIFT::create();
        case DESCRIPTOR_RGBSIFT:
            return RGBSIFT::create();
        case DESCRIPTOR_HoNC:
            return HoNC::create();
        default:
            // Print error message and return SIFT as default
            std::cerr << "Unknown descriptor type: " << type << ", using SIFT as default" << std::endl;
            if (max_features > 0) {
                return cv::SIFT::create(max_features);
            } else {
                return cv::SIFT::create();
            }
    }
}

void experiment_config::refreshDetectors() {
    
    // Recreate primary detector
    this->detector = createDescriptorExtractor(this->descriptorOptions.descriptorType, this->descriptorOptions.max_features);
    
    // Recreate secondary detector if stacking
    if(this->descriptorOptions.poolingStrategy == STACKING){
        this->detector2 = createDescriptorExtractor(this->descriptorOptions.descriptorType2, this->descriptorOptions.max_features);
        if(this->detector2) {
        } else {
            std::cerr << "[ERROR] Secondary detector recreation failed!" << std::endl;
        }
    } else {
        this->detector2.reset();  // Clear any existing detector2
    }
}

void experiment_config::verifyConfiguration(){

    if(this->verificationType != NO_VISUAL_VERIFICATION) {
        this->useMultiThreading = false;
    }

    // if descriptor type is RGBSIFT or HoNC then set the color space to D_COLOR
    if(this->descriptorOptions.descriptorType == DESCRIPTOR_HoNC ||
       this->descriptorOptions.descriptorType == DESCRIPTOR_RGBSIFT){
        this->descriptorOptions.descriptorColorSpace = D_COLOR;
    }

    if(this->descriptorOptions.poolingStrategy == STACKING){
        if(this->descriptorOptions.descriptorType2 == DESCRIPTOR_HoNC ||
           this->descriptorOptions.descriptorType2 == DESCRIPTOR_RGBSIFT){
            this->descriptorOptions.descriptorColorSpace = D_COLOR;
        }
    }


}



