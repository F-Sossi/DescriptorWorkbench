/*
 * Analysis Runner - CLI interface for Python analysis scripts
 * Bridges C++ experiment workflow with Python analysis tools
 */

#include <iostream>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <vector>

class AnalysisRunner {
private:
    std::string results_folder_;
    std::string output_folder_;
    std::string python_cmd_;

public:
    AnalysisRunner(const std::string& results_folder, const std::string& output_folder = "analysis/outputs")
        : results_folder_(results_folder), output_folder_(output_folder) {

        // Try to find Python 3
        if (system("python3 --version > /dev/null 2>&1") == 0) {
            python_cmd_ = "python3";
        } else if (system("python --version > /dev/null 2>&1") == 0) {
            python_cmd_ = "python";
        } else {
            throw std::runtime_error("Python not found. Please install Python 3.");
        }

        std::cout << "Using Python command: " << python_cmd_ << std::endl;
    }

    bool checkResultsFolder() const {
        if (!std::filesystem::exists(results_folder_)) {
            std::cerr << "❌ Results folder not found: " << results_folder_ << std::endl;
            return false;
        }

        // Check for CSV files
        bool has_csv = false;
        for (const auto& entry : std::filesystem::directory_iterator(results_folder_)) {
            if (entry.path().extension() == ".csv") {
                has_csv = true;
                break;
            }
        }

        if (!has_csv) {
            std::cerr << "⚠️ No CSV files found in results folder" << std::endl;
            return false;
        }

        std::cout << "✅ Results folder validated: " << results_folder_ << std::endl;
        return true;
    }

    bool checkPythonDependencies() const {
        std::cout << "Checking Python dependencies..." << std::endl;

        std::vector<std::string> required_packages = {
            "pandas", "numpy", "matplotlib", "seaborn"
        };

        for (const auto& package : required_packages) {
            std::string cmd = python_cmd_ + " -c \"import " + package + "\" 2>/dev/null";
            if (system(cmd.c_str()) != 0) {
                std::cerr << "❌ Missing Python package: " << package << std::endl;
                std::cerr << "Install with: pip install " << package << std::endl;
                return false;
            }
        }

        std::cout << "✅ All Python dependencies available" << std::endl;
        return true;
    }

    int runPrecisionRecallAnalysis(bool plots_only = false) const {
        std::cout << "\n=== Running Precision-Recall Analysis ===" << std::endl;

        std::string script_path = "analysis/scripts/precision_recall_analysis.py";
        if (!std::filesystem::exists(script_path)) {
            std::cerr << "❌ Analysis script not found: " << script_path << std::endl;
            return 1;
        }

        std::string cmd = python_cmd_ + " " + script_path + " \"" + results_folder_ + "\" --output \"" + output_folder_ + "\"";
        if (plots_only) {
            cmd += " --plots-only";
        }

        std::cout << "Running: " << cmd << std::endl;

        int result = system(cmd.c_str());
        if (result == 0) {
            std::cout << "✅ Precision-recall analysis completed successfully" << std::endl;
        } else {
            std::cerr << "❌ Precision-recall analysis failed with code: " << result << std::endl;
        }

        return result;
    }

    int generateReport() const {
        std::cout << "\n=== Generating Comprehensive Report ===" << std::endl;

        std::string script_path = "analysis/scripts/generate_report.py";
        if (!std::filesystem::exists(script_path)) {
            std::cerr << "❌ Report script not found: " << script_path << std::endl;
            return 1;
        }

        std::string cmd = python_cmd_ + " " + script_path + " \"" + results_folder_ + "\" --output \"" + output_folder_ + "\"";

        std::cout << "Running: " << cmd << std::endl;

        int result = system(cmd.c_str());
        if (result == 0) {
            std::cout << "✅ Report generation completed successfully" << std::endl;

            // Show output location
            std::string html_report = output_folder_ + "/analysis_report.html";
            if (std::filesystem::exists(html_report)) {
                std::cout << "📄 HTML Report: " << html_report << std::endl;
                std::cout << "💡 Open in browser to view detailed analysis" << std::endl;
            }
        } else {
            std::cerr << "❌ Report generation failed with code: " << result << std::endl;
        }

        return result;
    }

    void printUsage() const {
        std::cout << "\n=== Analysis Runner Usage ===" << std::endl;
        std::cout << "Available commands:" << std::endl;
        std::cout << "  --analysis     Run precision-recall analysis" << std::endl;
        std::cout << "  --plots-only   Generate only plots (faster)" << std::endl;
        std::cout << "  --report       Generate comprehensive HTML report" << std::endl;
        std::cout << "  --full         Run complete analysis pipeline" << std::endl;
        std::cout << "  --help         Show this help message" << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <results_folder> [options]" << std::endl;
        std::cerr << "       " << argv[0] << " --help" << std::endl;
        return 1;
    }

    std::string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h") {
        std::cout << "=== Analysis Runner ===" << std::endl;
        std::cout << "CLI tool for analyzing descriptor comparison results" << std::endl;
        std::cout << "\nUsage: " << argv[0] << " <results_folder> [options]" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  --analysis     Run precision-recall analysis" << std::endl;
        std::cout << "  --plots-only   Generate only plots (faster)" << std::endl;
        std::cout << "  --report       Generate comprehensive HTML report" << std::endl;
        std::cout << "  --full         Run complete analysis pipeline" << std::endl;
        return 0;
    }

    try {
        std::string results_folder = argv[1];
        AnalysisRunner runner(results_folder);

        // Validate setup
        if (!runner.checkResultsFolder()) {
            return 1;
        }

        if (!runner.checkPythonDependencies()) {
            std::cerr << "\n💡 Install missing dependencies with:" << std::endl;
            std::cerr << "   pip install -r analysis/requirements.txt" << std::endl;
            return 1;
        }

        // Parse command line options
        bool run_analysis = false;
        bool run_report = false;
        bool plots_only = false;
        bool run_full = false;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--analysis") {
                run_analysis = true;
            } else if (arg == "--report") {
                run_report = true;
            } else if (arg == "--plots-only") {
                plots_only = true;
                run_analysis = true;
            } else if (arg == "--full") {
                run_full = true;
            } else {
                std::cerr << "Unknown option: " << arg << std::endl;
                runner.printUsage();
                return 1;
            }
        }

        // Default action if no options specified
        if (!run_analysis && !run_report && !run_full) {
            std::cout << "No action specified. Running full analysis pipeline..." << std::endl;
            run_full = true;
        }

        int exit_code = 0;

        // Execute requested actions
        if (run_full) {
            std::cout << "🚀 Running complete analysis pipeline..." << std::endl;

            // Step 1: Run analysis
            if (runner.runPrecisionRecallAnalysis() != 0) {
                exit_code = 1;
            }

            // Step 2: Generate report
            if (exit_code == 0 && runner.generateReport() != 0) {
                exit_code = 1;
            }

            if (exit_code == 0) {
                std::cout << "\n🎉 Complete analysis pipeline finished successfully!" << std::endl;
            }
        } else {
            // Run individual components
            if (run_analysis) {
                if (runner.runPrecisionRecallAnalysis(plots_only) != 0) {
                    exit_code = 1;
                }
            }

            if (run_report && exit_code == 0) {
                if (runner.generateReport() != 0) {
                    exit_code = 1;
                }
            }
        }

        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
