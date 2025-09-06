#ifndef CORE_METRICS_METRICS_CALCULATOR_HPP
#define CORE_METRICS_METRICS_CALCULATOR_HPP

#include "ExperimentMetrics.hpp"
#include <vector>
#include <chrono>

/**
 * @brief Handles metrics calculation and aggregation for experiments
 * 
 * This class consolidates all metrics computation logic that was previously
 * scattered throughout the image processing pipeline.
 */
class MetricsCalculator {
public:
    /**
     * @brief Aggregate multiple folder metrics into overall experiment metrics
     * @param folder_metrics Vector of individual folder results
     * @param processing_time_ms Total processing time in milliseconds
     * @return Aggregated overall metrics
     */
    static ExperimentMetrics aggregateMetrics(const std::vector<ExperimentMetrics>& folder_metrics, 
                                             double processing_time_ms) {
        ExperimentMetrics overall_metrics;
        overall_metrics.success = true;
        overall_metrics.processing_time_ms = processing_time_ms;
        
        for (const auto& folder_metric : folder_metrics) {
            overall_metrics.merge(folder_metric);
        }
        
        // Calculate final metrics
        overall_metrics.calculateMeanPrecision();
        
        return overall_metrics;
    }

    /**
     * @brief Calculate processing time between two time points
     * @param start_time Starting time point
     * @param end_time Ending time point
     * @return Processing time in milliseconds
     */
    static double calculateProcessingTime(const std::chrono::high_resolution_clock::time_point& start_time,
                                        const std::chrono::high_resolution_clock::time_point& end_time) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        return duration.count();
    }

    /**
     * @brief Calculate precision from matches and correct matches
     * @param total_matches Total number of matches
     * @param correct_matches Number of correct matches
     * @return Precision value (0.0 to 1.0)
     */
    static double calculatePrecision(int total_matches, int correct_matches) {
        return total_matches > 0 ? static_cast<double>(correct_matches) / total_matches : 0.0;
    }

    /**
     * @brief Calculate precision from matches vector (assuming binary correct/incorrect)
     * @param matches Vector where each element indicates if match was correct
     * @param correct_matches_count Number of correct matches
     * @return Precision value (0.0 to 1.0)
     */
    static double calculatePrecisionFromMatches(const std::vector<bool>& matches, int correct_matches_count) {
        return matches.empty() ? 0.0 : static_cast<double>(correct_matches_count) / matches.size();
    }
};

#endif // CORE_METRICS_METRICS_CALCULATOR_HPP