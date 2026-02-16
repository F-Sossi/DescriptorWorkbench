#pragma once

#include "BenchmarkTypes.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace thesis_project::benchmark {

/**
 * @brief Formats benchmark results for display
 */
class ResultsFormatter {
public:
    /**
     * @brief Format results as a human-readable string
     */
    static std::string format(const Results& results) {
        std::ostringstream oss;

        auto pct = [](float v) {
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << (v * 100.0f) << "%";
            return s.str();
        };

        oss << "\n";
        oss << "========================================\n";
        oss << "HPatches Patch Benchmark Results\n";
        oss << "========================================\n";
        oss << "Descriptor: " << results.descriptor_name
            << " (" << results.descriptor_dimension << "D)\n";
        oss << "Scenes: " << results.num_scenes
            << ", Patches: " << results.num_patches << "\n";
        oss << "Time: " << std::fixed << std::setprecision(1)
            << results.processing_time_ms / 1000.0 << "s\n";
        oss << "----------------------------------------\n";

        // Matching results
        if (results.mAP_overall > 0.0f || results.accuracy_overall > 0.0f) {
            oss << "MATCHING\n";
            oss << "  Overall mAP:       " << pct(results.mAP_overall) << "\n";
            oss << "  Overall Accuracy:  " << pct(results.accuracy_overall) << "\n";
            oss << "  By Difficulty:\n";
            oss << "    Easy:   " << pct(results.mAP_easy) << "\n";
            oss << "    Hard:   " << pct(results.mAP_hard) << "\n";
            if (results.mAP_tough > 0) {
                oss << "    Tough:  " << pct(results.mAP_tough) << "\n";
            }
            oss << "  By Scene Type:\n";
            oss << "    Illumination: " << pct(results.mAP_illumination) << "\n";
            oss << "    Viewpoint:    " << pct(results.mAP_viewpoint) << "\n";
            oss << "  Detailed Breakdown:\n";
            oss << "    Illumination Easy: " << pct(results.mAP_illumination_easy) << "\n";
            oss << "    Illumination Hard: " << pct(results.mAP_illumination_hard) << "\n";
            oss << "    Viewpoint Easy:    " << pct(results.mAP_viewpoint_easy) << "\n";
            oss << "    Viewpoint Hard:    " << pct(results.mAP_viewpoint_hard) << "\n";
            oss << "----------------------------------------\n";
        }

        // Verification same-seq results
        if (results.verification_same_overall > 0.0f) {
            oss << "VERIFICATION (Same Sequence)\n";
            oss << "  Overall mAP: " << pct(results.verification_same_overall) << "\n";
            oss << "  By Difficulty:\n";
            oss << "    Easy:   " << pct(results.verification_same_easy) << "\n";
            oss << "    Hard:   " << pct(results.verification_same_hard) << "\n";
            if (results.verification_same_tough > 0) {
                oss << "    Tough:  " << pct(results.verification_same_tough) << "\n";
            }
            oss << "  By Scene Type:\n";
            oss << "    Illumination: " << pct(results.verification_same_illumination) << "\n";
            oss << "    Viewpoint:    " << pct(results.verification_same_viewpoint) << "\n";
            oss << "----------------------------------------\n";
        }

        // Verification diff-seq results
        if (results.verification_diff_overall > 0.0f) {
            oss << "VERIFICATION (Different Sequence)\n";
            oss << "  Overall mAP: " << pct(results.verification_diff_overall) << "\n";
            oss << "  By Difficulty:\n";
            oss << "    Easy:   " << pct(results.verification_diff_easy) << "\n";
            oss << "    Hard:   " << pct(results.verification_diff_hard) << "\n";
            if (results.verification_diff_tough > 0) {
                oss << "    Tough:  " << pct(results.verification_diff_tough) << "\n";
            }
            oss << "  By Scene Type:\n";
            oss << "    Illumination: " << pct(results.verification_diff_illumination) << "\n";
            oss << "    Viewpoint:    " << pct(results.verification_diff_viewpoint) << "\n";
            oss << "----------------------------------------\n";
        }

        // Retrieval results
        if (results.retrieval_overall > 0.0f) {
            oss << "RETRIEVAL\n";
            oss << "  Overall mAP: " << pct(results.retrieval_overall) << "\n";
            oss << "  By Difficulty:\n";
            oss << "    Easy:   " << pct(results.retrieval_easy) << "\n";
            oss << "    Hard:   " << pct(results.retrieval_hard) << "\n";
            if (results.retrieval_tough > 0) {
                oss << "    Tough:  " << pct(results.retrieval_tough) << "\n";
            }
            oss << "  By Scene Type:\n";
            oss << "    Illumination: " << pct(results.retrieval_illumination) << "\n";
            oss << "    Viewpoint:    " << pct(results.retrieval_viewpoint) << "\n";
            oss << "----------------------------------------\n";
        }

        oss << "========================================\n";

        return oss.str();
    }

    /**
     * @brief Print results to stdout
     */
    static void print(const Results& results) {
        std::cout << format(results);
    }
};

} // namespace thesis_project::benchmark
