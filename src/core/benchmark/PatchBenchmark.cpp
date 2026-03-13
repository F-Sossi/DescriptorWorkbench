#include "PatchBenchmark.hpp"
#include "PatchScope.hpp"
#include "DescriptorBank.hpp"
#include "tasks/MatchingTask.hpp"
#include "tasks/VerificationTask.hpp"
#include "tasks/RetrievalTask.hpp"
#include "core/patches/PatchLoader.hpp"
#include "thesis_project/database/DatabaseManager.hpp"
#include <chrono>
#include <iostream>

namespace thesis_project::benchmark {

Results PatchBenchmark::run(
    const Config& config,
    patches::IPatchDescriptorExtractor& extractor,
    const DescriptorParams& params,
    database::DatabaseManager* db,
    const ProgressCallback& progress) {

    auto start_time = std::chrono::high_resolution_clock::now();

    Results results;
    results.descriptor_name = extractor.name();
    results.descriptor_dimension = extractor.descriptorSize();

    // PHASE 1: Get scene list
    std::vector<std::string> scene_dirs;
    if (config.scenes.empty()) {
        scene_dirs = patches::PatchLoader::listScenes(config.patches_dir);
    } else {
        scene_dirs = config.scenes;
    }

    results.num_scenes = static_cast<int>(scene_dirs.size());

    if (config.verbose) {
        std::cout << "[PatchBenchmark] Running " << results.descriptor_name
                  << " (" << results.descriptor_dimension << "D)" << std::endl;
        std::cout << "[PatchBenchmark] Scenes: " << scene_dirs.size() << std::endl;
        std::cout << "[PatchBenchmark] Tasks: ";
        if (config.matching_enabled) std::cout << "matching ";
        if (config.verification_enabled) std::cout << "verification ";
        if (config.retrieval_enabled) std::cout << "retrieval ";
        std::cout << std::endl;
    }

    // PHASE 2: Build scope and extract descriptors
    if (config.verbose) {
        std::cout << "[PatchBenchmark] Building scope..." << std::endl;
    }

    PatchScope scope = PatchScope::build(config, scene_dirs);

    if (config.verbose) {
        std::cout << "[PatchBenchmark] Scope size: " << scope.size() << " patch sets" << std::endl;
    }

    DescriptorBank bank(scope);

    // Load from cache if enabled
    if (config.use_cached_descriptors && db && config.descriptor_cache_id > 0) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Loading cached descriptors..." << std::endl;
        }
        size_t loaded = bank.loadFromDatabase(*db, config.descriptor_cache_id);
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Loaded " << loaded << " descriptor sets from cache" << std::endl;
        }
    }

    // Extract missing descriptors
    if (bank.missingCount() > 0) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Extracting " << bank.missingCount()
                      << " missing descriptor sets..." << std::endl;
        }

        bank.extractMissing(config.patches_dir, config.color, extractor, params,
            [&](int current, int total, const std::string& msg) {
                if (progress) {
                    progress(current, total, "Extracting: " + msg);
                }
            });
    }

    results.num_patches = bank.totalPatches();

    // Store to cache if enabled
    if (config.store_descriptors_to_db && db && config.descriptor_cache_id > 0) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Storing descriptors to cache..." << std::endl;
        }
        bank.storeToDatabase(*db, config.descriptor_cache_id);
    }

    // PHASE 3: Run tasks
    if (config.verbose) {
        std::cout << std::endl;  // Newline after carriage-return progress output
    }

    if (config.matching_enabled) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Running matching task..." << std::endl;
        }

        auto matching = tasks::MatchingTask::run(bank, config, scene_dirs,
            [&](int current, int total, const std::string& scene) {
                if (progress) {
                    progress(current, total, "Matching: " + scene);
                }
            });

        mergeMatchingResults(results, matching);

        if (config.verbose) {
            std::cout << "\n[PatchBenchmark] Matching mAP: "
                      << (results.mAP_overall * 100.0f) << "%" << std::endl;
        }
    }

    if (config.verification_enabled) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Running verification task..." << std::endl;
        }

        auto verification = tasks::VerificationTask::run(bank, config);
        mergeVerificationResults(results, verification);

        if (config.verbose) {
            std::cout << "[PatchBenchmark] Verification (same-seq) mAP: "
                      << (results.verification_same_overall * 100.0f) << "%" << std::endl;
            std::cout << "[PatchBenchmark] Verification (diff-seq) mAP: "
                      << (results.verification_diff_overall * 100.0f) << "%" << std::endl;
        }
    }

    if (config.retrieval_enabled) {
        if (config.verbose) {
            std::cout << "[PatchBenchmark] Running retrieval task..." << std::endl;
        }

        auto retrieval = tasks::RetrievalTask::run(bank, config);
        mergeRetrievalResults(results, retrieval);

        if (config.verbose) {
            std::cout << "[PatchBenchmark] Retrieval mAP: "
                      << (results.retrieval_overall * 100.0f) << "%" << std::endl;
        }
    }

    // PHASE 4: Finalize
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    results.processing_time_ms = static_cast<double>(duration.count());

    if (config.print_results) {
        printResults(results);
    }

    return results;
}

void PatchBenchmark::mergeMatchingResults(Results& results, const tasks::MatchingResults& matching) {
    results.mAP_overall = matching.mAP_overall;
    results.accuracy_overall = matching.accuracy_overall;
    results.num_patches = matching.num_patches;

    results.mAP_easy = matching.mAP_easy;
    results.mAP_hard = matching.mAP_hard;
    results.mAP_tough = matching.mAP_tough;

    results.mAP_illumination = matching.mAP_illumination;
    results.mAP_viewpoint = matching.mAP_viewpoint;

    results.mAP_illumination_easy = matching.mAP_illumination_easy;
    results.mAP_illumination_hard = matching.mAP_illumination_hard;
    results.mAP_viewpoint_easy = matching.mAP_viewpoint_easy;
    results.mAP_viewpoint_hard = matching.mAP_viewpoint_hard;
}

void PatchBenchmark::mergeVerificationResults(Results& results, const tasks::VerificationResults& verification) {
    results.verification_same_overall = verification.same_seq_overall;
    results.verification_same_easy = verification.same_seq_easy;
    results.verification_same_hard = verification.same_seq_hard;
    results.verification_same_tough = verification.same_seq_tough;
    results.verification_same_illumination = verification.same_seq_illumination;
    results.verification_same_viewpoint = verification.same_seq_viewpoint;

    results.verification_diff_overall = verification.diff_seq_overall;
    results.verification_diff_easy = verification.diff_seq_easy;
    results.verification_diff_hard = verification.diff_seq_hard;
    results.verification_diff_tough = verification.diff_seq_tough;
    results.verification_diff_illumination = verification.diff_seq_illumination;
    results.verification_diff_viewpoint = verification.diff_seq_viewpoint;
}

void PatchBenchmark::mergeRetrievalResults(Results& results, const tasks::RetrievalResults& retrieval) {
    results.retrieval_overall = retrieval.mAP_overall;
    results.retrieval_easy = retrieval.mAP_easy;
    results.retrieval_hard = retrieval.mAP_hard;
    results.retrieval_tough = retrieval.mAP_tough;
    results.retrieval_illumination = retrieval.mAP_illumination;
    results.retrieval_viewpoint = retrieval.mAP_viewpoint;
}

} // namespace thesis_project::benchmark
