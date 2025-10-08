cmake_minimum_required(VERSION 3.16)
project(descriptor_compare)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures early (before any find_package calls that might use CUDA)
# RTX 4090 has compute capability 8.9
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

# Set CUDA compiler path if not already set
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/opt/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")
    endif()
endif()

# CUDA compatibility with newer GCC versions
if(DEFINED CMAKE_CUDA_COMPILER)
    # Force CUDA to use GCC 14 instead of the default GCC 15
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-14")

    # Use C++17 (same as main project)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Suppress deprecated GPU target warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

    message(STATUS "CUDA configured to use GCC 14 for compatibility")
endif()

# Build options
option(USE_CONAN "Use Conan for dependency management" OFF)
option(BUILD_PYTHON_BRIDGE "Build Python bridge for CNN descriptors" OFF)
option(USE_SYSTEM_PACKAGES "Prefer system packages over Conan" ON)
option(BUILD_LIBTORCH "Build LibTorch CNN descriptor support" ON)

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
endif()

# Architecture-specific optimizations (if supported)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Handle Conan vs System packages
if(USE_CONAN AND EXISTS ${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
    message(STATUS "Using Conan for dependency management")
else()
    message(STATUS "Using system packages for dependency management")
endif()

# Find packages with fallback support
find_package(PkgConfig QUIET)

# OpenCV - multiple detection methods
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(OpenCV REQUIRED opencv4)
    if(NOT OpenCV_FOUND)
        pkg_check_modules(OpenCV REQUIRED opencv)
    endif()
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or use Conan.")
endif()

message(STATUS "OpenCV found: ${OpenCV_VERSION}")

# Detect OpenCV contrib xfeatures2d availability to enable VGG wrapper at compile time
set(HAVE_XFEATURES2D OFF)
if(DEFINED OpenCV_LIBRARIES)
    string(JOIN ";" _OPENCV_LIBS ${OpenCV_LIBRARIES})
    string(TOLOWER "${_OPENCV_LIBS}" _OPENCV_LIBS_LOWER)
    if(_OPENCV_LIBS_LOWER MATCHES "xfeatures2d")
        set(HAVE_XFEATURES2D ON)
    endif()
endif()
if(HAVE_XFEATURES2D)
    add_compile_definitions(HAVE_OPENCV_XFEATURES2D)
    message(STATUS "OpenCV xfeatures2d detected: enabling VGG descriptor")
else()
    message(STATUS "OpenCV xfeatures2d not detected: VGG descriptor will be disabled")
endif()

# Boost
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem)
message(STATUS "Boost found: ${Boost_VERSION}")

# TBB
find_package(TBB QUIET)
if(NOT TBB_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(TBB tbb)
endif()

if(TBB_FOUND)
    message(STATUS "TBB found")
else()
    message(WARNING "TBB not found - multithreading may be limited")
endif()

# SQLite3 - for database storage
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(SQLite3 sqlite3)
endif()

if(NOT SQLite3_FOUND)
    # Try to find SQLite3 manually
    find_path(SQLite3_INCLUDE_DIR sqlite3.h)
    find_library(SQLite3_LIBRARY sqlite3)

    if(SQLite3_INCLUDE_DIR AND SQLite3_LIBRARY)
        set(SQLite3_FOUND TRUE)
        set(SQLite3_INCLUDE_DIRS ${SQLite3_INCLUDE_DIR})
        set(SQLite3_LIBRARIES ${SQLite3_LIBRARY})
        message(STATUS "SQLite3 found manually")
    else()
        message(FATAL_ERROR "SQLite3 not found. Please install SQLite3 development package.")
    endif()
else()
    message(STATUS "SQLite3 found: ${SQLite3_VERSION}")
endif()

# Google Test - for modern unit testing
if(PKG_CONFIG_FOUND)
    pkg_check_modules(GTEST gtest)
    pkg_check_modules(GTEST_MAIN gtest_main)
endif()

if(GTEST_FOUND AND GTEST_MAIN_FOUND)
    message(STATUS "Google Test found: using system packages")
    set(GTEST_LIBRARIES ${GTEST_LIBRARIES})
    set(GTEST_MAIN_LIBRARIES ${GTEST_MAIN_LIBRARIES})
    set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
else()
    message(STATUS "Google Test not found - legacy tests will continue to use manual main() functions")
endif()

# LibTorch - for CNN descriptor support
if(BUILD_LIBTORCH)
    # Set LibTorch root path for AUR installation
    set(CMAKE_PREFIX_PATH "/opt/libtorch;${CMAKE_PREFIX_PATH}")

    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "LibTorch found - enabling CNN descriptors")
        message(STATUS "LibTorch version: ${TORCH_VERSION}")
        message(STATUS "CUDA available: ${TORCH_CUDA_AVAILABLE}")
        set(BUILD_LIBTORCH_DESCRIPTORS ON)
    else()
        message(STATUS "LibTorch not found - CNN descriptors disabled")
        message(STATUS "To enable: install libtorch or libtorch-cuda from AUR")
        set(BUILD_LIBTORCH_DESCRIPTORS OFF)
    endif()
else()
    message(STATUS "LibTorch support disabled (use -DBUILD_LIBTORCH=ON to enable)")
    set(BUILD_LIBTORCH_DESCRIPTORS OFF)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

if(USE_CONAN)
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
    if(TBB_FOUND)
        include_directories(${TBB_INCLUDE_DIRS})
    endif()
    include_directories(${SQLite3_INCLUDE_DIRS})
endif()

# Add Google Test include directories if found
if(GTEST_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

# Link directories
if(USE_CONAN)
    link_directories(${OpenCV_LIB_DIRS})
    link_directories(${Boost_LIB_DIRS})
    if(TBB_FOUND)
        link_directories(${TBB_LIB_DIRS})
    endif()
    link_directories(${SQLite3_LIB_DIRS})
endif()

# ================================
# EXISTING LIBRARIES
# ================================

# KEYPOINTS LIBRARY
set(KEYPOINTS_SOURCES
    keypoints/VanillaSIFT.cpp
    keypoints/DSPSIFT.cpp
    keypoints/RGBSIFT.cpp
    keypoints/HoNC.cpp
    keypoints/HoWH.cpp
)

add_library(keypoints ${KEYPOINTS_SOURCES})
target_include_directories(keypoints PUBLIC keypoints)

if(USE_CONAN)
    target_link_libraries(keypoints ${OpenCV_LIBS})
else()
    target_link_libraries(keypoints ${OpenCV_LIBRARIES})
endif()

# DATABASE LIBRARY (LEGACY - REMOVED)
# Legacy database implementation removed in favor of src/core/database/DatabaseManager.cpp
# The modern implementation is in thesis_database library

# STAGE 5: NEW DATABASE LIBRARY (THESIS PROJECT)
if(EXISTS "${CMAKE_SOURCE_DIR}/src/core/database/DatabaseManager.cpp")
    add_library(thesis_database
        src/core/database/DatabaseManager.cpp
    )

    target_include_directories(thesis_database PUBLIC
        include
        ${SQLite3_INCLUDE_DIRS}
    )

    target_link_libraries(thesis_database
        ${SQLite3_LIBRARIES}
        ${OpenCV_LIBS}
    )

    target_compile_features(thesis_database PRIVATE cxx_std_17)
    message(STATUS "Stage 5: Thesis database library configured")
else()
    message(FATAL_ERROR "Stage 5: DatabaseManager.cpp not found; database integration is required")
endif()

# ================================
# MAIN APPLICATION
# ================================

# Legacy descriptor_compare executable removed - use experiment_runner CLI instead

# Database integration moved to CLI tools (experiment_runner, keypoint_manager)

# PATH DEFINITIONS (kept for compatibility with other targets)
set(DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/")
set(RESULTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/results/")
set(KEYPOINTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/reference_keypoints/")
set(DATABASE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/experiments.db")

# Create necessary directories
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reference_keypoints)

# ================================
# COMPREHENSIVE TESTING SYSTEM (CLEAN)
# ================================

# Enable testing
enable_testing()

# Helper function to safely create test executables
function(create_test_if_exists test_source test_name)
    if(EXISTS "${CMAKE_SOURCE_DIR}/${test_source}")
        add_executable(${test_name} ${test_source})
        target_compile_features(${test_name} PRIVATE cxx_std_17)

        # Link libraries based on test type
        if(${test_name} MATCHES "database")
            if(TARGET thesis_database)
                target_link_libraries(${test_name} thesis_database)
            endif()
            if(${test_name} MATCHES "integration" AND NOT ${test_name} MATCHES "simple")
                target_link_libraries(${test_name} keypoints)
                target_include_directories(${test_name} PRIVATE descriptor_compare keypoints)
            endif()
        elseif(${test_name} MATCHES "interface|headers")
            target_link_libraries(${test_name} keypoints)
            target_include_directories(${test_name} PRIVATE keypoints descriptor_compare)
        endif()

        # Add to CTest
        add_test(NAME ${test_name} COMMAND ${test_name})

        # Set test labels
        if(${test_name} MATCHES "database")
            set_tests_properties(${test_name} PROPERTIES LABELS "database;stage5" TIMEOUT 60)
        elseif(${test_name} MATCHES "interface|headers")
            set_tests_properties(${test_name} PROPERTIES LABELS "interface;stage3" TIMEOUT 60)
        elseif(${test_name} MATCHES "types")
            set_tests_properties(${test_name} PROPERTIES LABELS "types;stage2" TIMEOUT 60)
        elseif(${test_name} MATCHES "yaml|config")
            set_tests_properties(${test_name} PROPERTIES LABELS "config;stage4" TIMEOUT 60)
        else()
            set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
        endif()

        message(STATUS "Test configured: ${test_name}")
    else()
        message(STATUS "Test source not found (skipping): ${test_source}")
    endif()
endfunction()

# Helper function to create Google Test executables
function(create_gtest_if_exists test_source test_name)
    if(NOT GTEST_FOUND OR NOT GTEST_MAIN_FOUND)
        message(STATUS "Google Test not available - skipping: ${test_name}")
        return()
    endif()
    
    if(EXISTS "${CMAKE_SOURCE_DIR}/${test_source}")
        add_executable(${test_name} ${test_source})
        target_compile_features(${test_name} PRIVATE cxx_std_17)
        
        # Link Google Test libraries
        target_link_libraries(${test_name} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
        
        # Link libraries based on test type (same logic as legacy tests)
        if(${test_name} MATCHES "database")
            if(TARGET thesis_database)
                target_link_libraries(${test_name} thesis_database)
            endif()
            if(${test_name} MATCHES "integration" AND NOT ${test_name} MATCHES "simple")
                target_link_libraries(${test_name} keypoints)
                target_include_directories(${test_name} PRIVATE descriptor_compare keypoints)
            endif()
        elseif(${test_name} MATCHES "interface|headers")
            target_link_libraries(${test_name} keypoints)
            target_include_directories(${test_name} PRIVATE keypoints descriptor_compare)
        elseif(${test_name} MATCHES "stage7")
            # Stage 7 tests need OpenCV
            target_link_libraries(${test_name} ${OpenCV_LIBRARIES})
            target_include_directories(${test_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
            
            # Stage 7 migration tests need additional dependencies
            if(${test_name} MATCHES "migration")
                target_link_libraries(${test_name} keypoints)
                target_include_directories(${test_name} PRIVATE 
                    src 
                    keypoints 
                    descriptor_compare
                )
                # Also add pooling and legacy config sources used by migration path
                target_sources(${test_name} PRIVATE 
                    src/core/pooling/PoolingFactory.cpp
                    src/core/pooling/NoPooling.cpp
                    src/core/pooling/DomainSizePooling.cpp
                    src/core/pooling/StackingPooling.cpp
                    src/core/matching/BruteForceMatching.cpp
                    src/core/matching/FLANNMatching.cpp
                    src/core/matching/MatchingFactory.cpp
                    src/core/processing/processor_utils.cpp
                )
            endif()
        elseif(${test_name} MATCHES "metrics|true_average_precision")
            # Metrics tests need OpenCV for TrueAveragePrecision.hpp and the implementation
            target_sources(${test_name} PRIVATE src/core/metrics/TrueAveragePrecision.cpp)
            target_link_libraries(${test_name} ${OpenCV_LIBRARIES})
            target_include_directories(${test_name} PRIVATE ${OpenCV_INCLUDE_DIRS} src)
        elseif(${test_name} MATCHES "matching")
            # Matching tests need matching sources and OpenCV
            target_sources(${test_name} PRIVATE
                src/core/matching/BruteForceMatching.cpp
                src/core/matching/FLANNMatching.cpp
                src/core/matching/MatchingFactory.cpp
            )
            target_link_libraries(${test_name} ${OpenCV_LIBRARIES} keypoints)
            target_include_directories(${test_name} PRIVATE ${OpenCV_INCLUDE_DIRS} src descriptor_compare keypoints)
        elseif(${test_name} MATCHES "pooling")
            # Pooling tests need OpenCV, keypoints, and pooling source files
        target_sources(${test_name} PRIVATE 
                src/core/pooling/PoolingFactory.cpp
                src/core/pooling/NoPooling.cpp
                src/core/pooling/DomainSizePooling.cpp
                src/core/pooling/StackingPooling.cpp
                       src/core/descriptor/factories/DescriptorFactory.cpp
                       src/core/descriptor/extractors/wrappers/SIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/RGBSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/HoNCWrapper.cpp
                       src/core/descriptor/extractors/wrappers/VSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/DSPSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.cpp
                       src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.cpp
                       src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.cpp
                       src/core/descriptor/extractors/wrappers/VGGWrapper.cpp
                       src/core/descriptor/extractors/wrappers/ORBWrapper.cpp
                       src/core/descriptor/extractors/wrappers/SURFWrapper.cpp
            )
            target_link_libraries(${test_name} ${OpenCV_LIBRARIES} keypoints)
            target_include_directories(${test_name} PRIVATE ${OpenCV_INCLUDE_DIRS} src keypoints descriptor_compare)
        endif()
        
        # Add to CTest
        add_test(NAME ${test_name} COMMAND ${test_name})
        
        # Set test labels with gtest suffix and proper working directory
        if(${test_name} MATCHES "database")
            set_tests_properties(${test_name} PROPERTIES LABELS "database;stage5;gtest" TIMEOUT 60)
        elseif(${test_name} MATCHES "interface|headers")
            set_tests_properties(${test_name} PROPERTIES LABELS "interface;stage3;gtest" TIMEOUT 60)
        elseif(${test_name} MATCHES "types")
            set_tests_properties(${test_name} PROPERTIES LABELS "types;stage2;gtest" TIMEOUT 60)
        elseif(${test_name} MATCHES "yaml|config")
            set_tests_properties(${test_name} PROPERTIES 
                LABELS "config;stage4;gtest" 
                TIMEOUT 60 
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        else()
            set_tests_properties(${test_name} PROPERTIES LABELS "gtest" TIMEOUT 60)
        endif()
        
        message(STATUS "Google Test configured: ${test_name}")
    else()
        message(STATUS "Google Test source not found (skipping): ${test_source}")
    endif()
endfunction()

# Legacy manual tests have been migrated to Google Test framework
# All remaining manual test files use create_test_if_exists for backward compatibility

# Google Test configuration tests
create_gtest_if_exists("tests/unit/test_yaml_config_gtest.cpp" "test_yaml_config_gtest")

# Google Test basic component tests
create_gtest_if_exists("tests/unit/test_constants_gtest.cpp" "test_constants_gtest")
create_gtest_if_exists("tests/unit/test_new_types_gtest.cpp" "test_new_types_gtest")
create_gtest_if_exists("tests/unit/simple_interface_test_gtest.cpp" "simple_interface_test_gtest")

# Google Test metrics calculation tests (Phase 1 comprehensive testing)
create_gtest_if_exists("tests/unit/metrics/test_metrics_calculator_gtest.cpp" "test_metrics_calculator_gtest")
create_gtest_if_exists("tests/unit/metrics/test_true_average_precision_gtest.cpp" "test_true_average_precision_gtest")
create_gtest_if_exists("tests/unit/metrics/test_experiment_metrics_gtest.cpp" "test_experiment_metrics_gtest")

# Google Test pooling strategy tests (Phase 2 comprehensive testing)
create_gtest_if_exists("tests/unit/pooling/test_pooling_factory_gtest.cpp" "test_pooling_factory_gtest")

# Experiment helper tests
create_gtest_if_exists("tests/unit/experiment/test_experiment_helpers_gtest.cpp" "test_experiment_helpers_gtest")
if(TARGET test_experiment_helpers_gtest)
    target_sources(test_experiment_helpers_gtest PRIVATE
        src/core/experiment/ExperimentHelpers.cpp
        src/core/matching/BruteForceMatching.cpp
        src/core/metrics/TrueAveragePrecision.cpp)
    target_link_libraries(test_experiment_helpers_gtest ${OpenCV_LIBRARIES})
    target_include_directories(test_experiment_helpers_gtest PRIVATE src include ${OpenCV_INCLUDE_DIRS})
endif()

# Google Test matching factory tests
create_gtest_if_exists("tests/unit/factories/test_matching_factory_gtest.cpp" "test_matching_factory_gtest")
if(TARGET test_matching_factory_gtest)
    target_sources(test_matching_factory_gtest PRIVATE 
        src/core/matching/BruteForceMatching.cpp
        src/core/matching/FLANNMatching.cpp
        src/core/matching/RatioTestMatching.cpp
        src/core/matching/MatchingFactory.cpp
    )
    target_link_libraries(test_matching_factory_gtest ${OpenCV_LIBRARIES} keypoints)
    target_include_directories(test_matching_factory_gtest PRIVATE src include descriptor_compare keypoints ${OpenCV_INCLUDE_DIRS})
endif()

# Google Test descriptor factory tests
create_gtest_if_exists("tests/unit/factories/test_descriptor_factory_gtest.cpp" "test_descriptor_factory_gtest")
if(TARGET test_descriptor_factory_gtest)
    target_sources(test_descriptor_factory_gtest PRIVATE 
        src/core/descriptor/factories/DescriptorFactory.cpp
        src/core/descriptor/extractors/wrappers/SIFTWrapper.cpp
        src/core/descriptor/extractors/wrappers/RGBSIFTWrapper.cpp
        src/core/descriptor/extractors/wrappers/HoNCWrapper.cpp
        src/core/descriptor/extractors/wrappers/VSIFTWrapper.cpp
        src/core/descriptor/extractors/wrappers/DSPSIFTWrapper.cpp
        src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.cpp
        src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.cpp
        src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.cpp
        src/core/descriptor/extractors/wrappers/VGGWrapper.cpp
        src/core/descriptor/extractors/wrappers/ORBWrapper.cpp
        src/core/descriptor/extractors/wrappers/SURFWrapper.cpp)
    target_link_libraries(test_descriptor_factory_gtest keypoints ${OpenCV_LIBRARIES})
    target_include_directories(test_descriptor_factory_gtest PRIVATE src include keypoints descriptor_compare ${OpenCV_INCLUDE_DIRS})
endif()

# Google Test DSP color wrapper smoke tests
create_gtest_if_exists("tests/unit/descriptors/test_dsp_color_wrappers_gtest.cpp" "test_dsp_color_wrappers_gtest")
if(TARGET test_dsp_color_wrappers_gtest)
    target_sources(test_dsp_color_wrappers_gtest PRIVATE
        src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.cpp
        src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.cpp
        src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.cpp)
    target_link_libraries(test_dsp_color_wrappers_gtest keypoints ${OpenCV_LIBRARIES})
    target_include_directories(test_dsp_color_wrappers_gtest PRIVATE src include keypoints descriptor_compare ${OpenCV_INCLUDE_DIRS})
endif()

# Google Test keypoint generator tests (Interface upgrade completion)
create_gtest_if_exists("tests/unit/keypoints/test_keypoint_generators.cpp" "test_keypoint_generators_gtest")
if(TARGET test_keypoint_generators_gtest)
    target_sources(test_keypoint_generators_gtest PRIVATE 
        src/core/keypoints/detectors/SIFTKeypointGenerator.cpp
        src/core/keypoints/detectors/HarrisKeypointGenerator.cpp
        src/core/keypoints/detectors/ORBKeypointGenerator.cpp
        src/core/keypoints/detectors/NonOverlappingKeypointGenerator.cpp
        src/core/keypoints/KeypointGeneratorFactory.cpp
        src/core/keypoints/generators/KeynetDetector.cpp)
    target_link_libraries(test_keypoint_generators_gtest ${OpenCV_LIBRARIES})
    target_include_directories(test_keypoint_generators_gtest PRIVATE src include ${OpenCV_INCLUDE_DIRS})
endif()

## Removed: Stage 7 integration and bridge tests (legacy)

## Removed: YAML loader + bridge tests (legacy ConfigurationBridge)

## Removed: processor utils migration tests (legacy)

# Database tests (in database subdirectory) - legacy manual tests migrated to Google Test
## Removed: legacy database integration manual test

# Google Test database tests
create_gtest_if_exists("tests/unit/database/test_database_gtest.cpp" "test_database_gtest")
create_gtest_if_exists("tests/unit/database/test_database_integration_simple_gtest.cpp" "test_database_integration_simple_gtest")
create_gtest_if_exists("tests/unit/database/test_database_standalone_gtest.cpp" "test_database_standalone_gtest")

# Custom targets for running test groups
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    COMMENT "Running all tests"
)

add_custom_target(run_database_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L database --output-on-failure --verbose
    COMMENT "Running database tests only"
)

add_custom_target(run_interface_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L interface --output-on-failure --verbose
    COMMENT "Running interface tests only"
)

add_custom_target(run_config_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L config --output-on-failure --verbose
    COMMENT "Running configuration tests only"
)

add_custom_target(run_stage_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L "stage2|stage3|stage4|stage5" --output-on-failure --verbose
    COMMENT "Running all stage tests"
)

add_custom_target(run_gtest_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L gtest --output-on-failure --verbose
    COMMENT "Running Google Test framework tests only"
)

# Install rules - descriptor_compare removed, use experiment_runner instead
install(DIRECTORY config/ DESTINATION share/thesis_project/config
        FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml")

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Use Conan: ${USE_CONAN}")
message(STATUS "  Use system packages: ${USE_SYSTEM_PACKAGES}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  Boost: ${Boost_VERSION}")
message(STATUS "  TBB: ${TBB_FOUND}")
message(STATUS "  SQLite3: ${SQLite3_FOUND}")
message(STATUS "")
message(STATUS "Testing:")
message(STATUS "  Available test targets: run_all_tests, run_database_tests, run_interface_tests, run_config_tests, run_gtest_tests")
if(GTEST_FOUND AND GTEST_MAIN_FOUND)
    message(STATUS "  Google Test: ENABLED - Modern test framework available")
else()
    message(STATUS "  Google Test: DISABLED - Using legacy manual tests")
endif()
message(STATUS "")

# ================================
# STAGE 6: ANALYSIS INTEGRATION
# ================================

# Analysis CLI runner (optional component)
option(BUILD_ANALYSIS "Build analysis integration" ON)

if(BUILD_ANALYSIS)
    message(STATUS "Building Stage 6 analysis integration")

    # Check if analysis runner exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/cli/analysis_runner.cpp")
        add_executable(analysis_runner cli/analysis_runner.cpp)
        target_compile_features(analysis_runner PRIVATE cxx_std_17)

        # No special libraries needed for basic analysis runner
        # It calls Python scripts via system()

        message(STATUS "Analysis runner configured")

        # Create custom target for running analysis
        add_custom_target(run_analysis
            COMMAND ./analysis_runner "${CMAKE_SOURCE_DIR}/results" --full
            DEPENDS analysis_runner
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running complete analysis pipeline on results folder"
        )

        add_custom_target(quick_analysis
            COMMAND ./analysis_runner "${CMAKE_SOURCE_DIR}/results" --plots-only
            DEPENDS analysis_runner
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running quick analysis (plots only)"
        )

        message(STATUS "Analysis targets configured: run_analysis, quick_analysis")

    else()
        message(WARNING "analysis_runner.cpp not found - skipping analysis integration")
    endif()
else()
    message(STATUS "Analysis integration disabled (use -DBUILD_ANALYSIS=ON to enable)")
endif()

# ================================
# CLI TOOLS
# ================================

# CLI experiment runner using YAML configuration
option(BUILD_EXPERIMENT_CLI "Build CLI experiment runner with YAML support" ON)

# Keypoint manager CLI tool
option(BUILD_KEYPOINT_MANAGER "Build keypoint manager CLI tool" ON)

if(BUILD_EXPERIMENT_CLI)
    message(STATUS "Building CLI experiment runner")

    # Check if experiment runner exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/cli/experiment_runner.cpp")
        add_executable(experiment_runner
                       cli/experiment_runner.cpp
                       src/core/config/YAMLConfigLoader.cpp
                       src/core/pooling/NoPooling.cpp
                       src/core/pooling/DomainSizePooling.cpp
                       src/core/pooling/StackingPooling.cpp
                       src/core/pooling/PoolingFactory.cpp
                       src/core/matching/BruteForceMatching.cpp
                       src/core/matching/FLANNMatching.cpp
                       src/core/matching/RatioTestMatching.cpp
                       src/core/matching/MatchingFactory.cpp
                       src/core/metrics/TrueAveragePrecision.cpp
                       src/core/experiment/ExperimentHelpers.cpp
                       src/core/descriptor/factories/DescriptorFactory.cpp
                       src/core/descriptor/extractors/wrappers/SIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/RGBSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/HoNCWrapper.cpp
                       src/core/descriptor/extractors/wrappers/VSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/DSPSIFTWrapper.cpp
                       src/core/keypoints/KeypointAttributeAdapter.cpp)
        # DNN wrapper
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/DNNPatchWrapper.cpp)
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/PseudoDNNWrapper.cpp)
        # VGG wrapper (requires OpenCV contrib xfeatures2d)
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/VGGWrapper.cpp)
        # ORB wrapper
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/ORBWrapper.cpp)
        # SURF wrapper
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/SURFWrapper.cpp)
        # DSP V2 wrappers for color descriptors
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.cpp)
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.cpp)
        target_sources(experiment_runner PRIVATE src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.cpp)

        # LibTorch wrapper (if available)
        if(BUILD_LIBTORCH_DESCRIPTORS)
            target_sources(experiment_runner PRIVATE
                src/core/descriptor/extractors/wrappers/LibTorchWrapper.cpp
                src/core/descriptor/extractors/wrappers/LibTorchFactory.cpp)
            target_link_libraries(experiment_runner ${TORCH_LIBRARIES})
            target_compile_definitions(experiment_runner PRIVATE BUILD_LIBTORCH_DESCRIPTORS=1)
            message(STATUS "LibTorch wrapper added to experiment_runner")
        endif()
        
        target_compile_features(experiment_runner PRIVATE cxx_std_17)
        
        # Include directories
        target_include_directories(experiment_runner PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/descriptor_compare
        )

        # Path definitions (same as main descriptor_compare)
        target_compile_definitions(experiment_runner PRIVATE
            DATA_PATH="${DATA_PATH}"
            RESULTS_PATH="${RESULTS_PATH}"
            KEYPOINTS_PATH="${KEYPOINTS_PATH}"
            DATABASE_PATH="${DATABASE_PATH}"
        )

        # Link libraries
        target_link_libraries(experiment_runner
            keypoints
            Boost::system
            Boost::filesystem
        )

        target_link_libraries(experiment_runner thesis_database)

        # Add OpenCV and other dependencies
        if(USE_CONAN)
            target_link_libraries(experiment_runner ${OpenCV_LIBS})
            target_link_libraries(experiment_runner ${SQLite3_LIBS})
        else()
            target_link_libraries(experiment_runner ${OpenCV_LIBRARIES})
            target_link_libraries(experiment_runner ${SQLite3_LIBRARIES})
        endif()

        # YAML-cpp dependency
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(YAMLCPP yaml-cpp)
        endif()
        
        if(YAMLCPP_FOUND)
            target_link_libraries(experiment_runner ${YAMLCPP_LIBRARIES})
            target_include_directories(experiment_runner PRIVATE ${YAMLCPP_INCLUDE_DIRS})
            message(STATUS "yaml-cpp found via pkg-config for experiment runner")
        else()
            # Try system yaml-cpp
            target_link_libraries(experiment_runner yaml-cpp)
            message(STATUS "Using system yaml-cpp library for experiment runner")
        endif()

        # OpenMP for scene-level parallelism (optional but recommended)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(experiment_runner OpenMP::OpenMP_CXX)
            message(STATUS "OpenMP enabled for parallel scene processing (configurable via YAML)")
        else()
            message(WARNING "OpenMP not found - experiment_runner will run sequentially")
        endif()

        message(STATUS "CLI experiment runner configured")
        
    else()
        message(WARNING "experiment_runner.cpp not found - skipping CLI experiment runner")
    endif()
else()
    message(STATUS "CLI experiment runner disabled (use -DBUILD_EXPERIMENT_CLI=ON to enable)")
endif()

# Keypoint manager CLI tool
if(BUILD_KEYPOINT_MANAGER)
    message(STATUS "Building keypoint manager CLI tool")
    
    # Check if keypoint manager exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/cli/keypoint_manager.cpp")
        add_executable(keypoint_manager
                       cli/keypoint_manager.cpp
                       src/core/keypoints/locked_in_keypoints.cpp
                       src/core/keypoints/KeypointGeneratorFactory.cpp
                       src/core/keypoints/KeypointAttributeAdapter.cpp
                       src/core/keypoints/detectors/SIFTKeypointGenerator.cpp
                       src/core/keypoints/detectors/HarrisKeypointGenerator.cpp
                       src/core/keypoints/detectors/ORBKeypointGenerator.cpp
                       src/core/keypoints/detectors/NonOverlappingKeypointGenerator.cpp
                       src/core/keypoints/generators/KeynetDetector.cpp
                       src/core/processing/processor_utils.cpp
                       src/core/pooling/NoPooling.cpp
                       src/core/pooling/DomainSizePooling.cpp
                       src/core/pooling/StackingPooling.cpp
                       src/core/pooling/PoolingFactory.cpp
                       src/core/matching/BruteForceMatching.cpp
                       src/core/matching/FLANNMatching.cpp
                       src/core/matching/RatioTestMatching.cpp
                       src/core/matching/MatchingFactory.cpp
                       src/core/descriptor/factories/DescriptorFactory.cpp
                       src/core/descriptor/extractors/wrappers/SIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/RGBSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/HoNCWrapper.cpp
                       src/core/descriptor/extractors/wrappers/VSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/DSPSIFTWrapper.cpp
                       src/core/descriptor/extractors/wrappers/DSPRGBSIFTWrapperV2.cpp
                       src/core/descriptor/extractors/wrappers/DSPHoWHWrapperV2.cpp
                       src/core/descriptor/extractors/wrappers/DSPHoNCWrapperV2.cpp)

        # VGG wrapper (requires OpenCV contrib xfeatures2d)
        target_sources(keypoint_manager PRIVATE src/core/descriptor/extractors/wrappers/VGGWrapper.cpp)
        # ORB wrapper
        target_sources(keypoint_manager PRIVATE src/core/descriptor/extractors/wrappers/ORBWrapper.cpp)
        # SURF wrapper
        target_sources(keypoint_manager PRIVATE src/core/descriptor/extractors/wrappers/SURFWrapper.cpp)
        
        target_compile_features(keypoint_manager PRIVATE cxx_std_17)
        
        # Include directories
        target_include_directories(keypoint_manager PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/descriptor_compare
        )

        # Path definitions
        target_compile_definitions(keypoint_manager PRIVATE
            DATA_PATH="${DATA_PATH}"
            RESULTS_PATH="${RESULTS_PATH}"
            KEYPOINTS_PATH="${KEYPOINTS_PATH}"
            DATABASE_PATH="${DATABASE_PATH}"
        )

        # Link libraries
        target_link_libraries(keypoint_manager
            keypoints
            thesis_database
            Boost::system
            Boost::filesystem
        )

        # Add OpenCV and other dependencies
        if(USE_CONAN)
            target_link_libraries(keypoint_manager ${OpenCV_LIBS})
            target_link_libraries(keypoint_manager ${SQLite3_LIBS})
        else()
            target_link_libraries(keypoint_manager ${OpenCV_LIBRARIES})
            target_link_libraries(keypoint_manager ${SQLite3_LIBRARIES})
        endif()

        message(STATUS "Keypoint manager CLI tool configured")
        
    else()
        message(WARNING "keypoint_manager.cpp not found - skipping keypoint manager CLI")
    endif()
else()
    message(STATUS "Keypoint manager disabled (use -DBUILD_KEYPOINT_MANAGER=ON to enable)")
endif()

# Stage 7 migration system removed: new pipeline is the default
# YAML negative validation tests
create_gtest_if_exists("tests/unit/config/test_yaml_validation_errors_gtest.cpp" "test_yaml_validation_errors_gtest")
if(TARGET test_yaml_validation_errors_gtest)
    target_sources(test_yaml_validation_errors_gtest PRIVATE src/core/config/YAMLConfigLoader.cpp)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(YAMLCPP yaml-cpp)
    endif()
    if(YAMLCPP_FOUND)
        target_link_libraries(test_yaml_validation_errors_gtest ${YAMLCPP_LIBRARIES})
        target_include_directories(test_yaml_validation_errors_gtest PRIVATE ${YAMLCPP_INCLUDE_DIRS})
    else()
        target_link_libraries(test_yaml_validation_errors_gtest yaml-cpp)
    endif()
    target_link_libraries(test_yaml_validation_errors_gtest ${OpenCV_LIBRARIES})
    target_include_directories(test_yaml_validation_errors_gtest PRIVATE src include ${OpenCV_INCLUDE_DIRS})

# YAML Schema v1 tests
create_gtest_if_exists("tests/unit/config/test_yaml_schema_v1_gtest.cpp" "test_yaml_schema_v1_gtest")
if(TARGET test_yaml_schema_v1_gtest)
    target_sources(test_yaml_schema_v1_gtest PRIVATE src/core/config/YAMLConfigLoader.cpp)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(YAMLCPP yaml-cpp)
    endif()
    if(YAMLCPP_FOUND)
        target_link_libraries(test_yaml_schema_v1_gtest ${YAMLCPP_LIBRARIES})
        target_include_directories(test_yaml_schema_v1_gtest PRIVATE ${YAMLCPP_INCLUDE_DIRS})
    else()
        target_link_libraries(test_yaml_schema_v1_gtest yaml-cpp)
    endif()
    target_link_libraries(test_yaml_schema_v1_gtest ${OpenCV_LIBRARIES})
    target_include_directories(test_yaml_schema_v1_gtest PRIVATE src include ${OpenCV_INCLUDE_DIRS})
endif()
endif()
