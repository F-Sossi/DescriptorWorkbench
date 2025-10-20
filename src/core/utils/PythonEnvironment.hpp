#pragma once

#include <string>
#include <vector>
#include <optional>

namespace thesis_project {
namespace utils {

/**
 * @brief Python environment detection and command generation utility
 *
 * This class provides portable Python environment detection that works across:
 * - Conda environments (miniforge, anaconda, miniconda)
 * - Virtual environments (venv, virtualenv)
 * - System Python installations
 * - Docker containers
 *
 * It automatically detects available Python installations, validates required
 * packages, and generates appropriate activation commands for running Python scripts.
 */
class PythonEnvironment {
public:
    /**
     * @brief Python environment type
     */
    enum class Type {
        CONDA,          // Conda environment (detected by CONDA_PREFIX)
        VIRTUALENV,     // Virtual environment (detected by VIRTUAL_ENV)
        SYSTEM,         // System Python (no environment detected)
        UNKNOWN         // Could not determine or Python not found
    };

    /**
     * @brief Configuration for Python environment requirements
     */
    struct Requirements {
        std::vector<std::string> packages;  // Required Python packages (e.g., "kornia", "torch")
        std::string python_version;          // Minimum Python version (e.g., "3.8")

        Requirements() = default;
        Requirements(const std::vector<std::string>& pkgs) : packages(pkgs) {}
    };

    /**
     * @brief Detected Python environment information
     */
    struct Info {
        Type type = Type::UNKNOWN;
        std::string python_path;             // Path to Python executable
        std::string environment_name;        // Name of conda/venv environment (if applicable)
        std::string environment_path;        // Path to environment root
        std::string activation_prefix;       // Shell command prefix for activation
        bool has_required_packages = false;  // Whether all required packages are available
        std::vector<std::string> missing_packages; // List of missing packages

        bool is_valid() const {
            return type != Type::UNKNOWN && !python_path.empty() && has_required_packages;
        }
    };

    /**
     * @brief Detect Python environment and validate requirements
     *
     * This method performs automatic detection in the following order:
     * 1. Active conda environment (via CONDA_PREFIX env var)
     * 2. Active virtual environment (via VIRTUAL_ENV env var)
     * 3. System Python (via 'which python3' or 'which python')
     *
     * @param requirements Required packages and Python version
     * @return Detected environment information
     */
    static Info detect(const Requirements& requirements = {});

    /**
     * @brief Generate shell command to run Python script with proper environment activation
     *
     * @param script_path Path to Python script to execute
     * @param args Additional command-line arguments for the script
     * @param info Pre-detected environment info (if not provided, auto-detects)
     * @return Complete shell command string with activation and execution
     */
    static std::string generateCommand(
        const std::string& script_path,
        const std::vector<std::string>& args = {},
        const std::optional<Info>& info = std::nullopt
    );

    /**
     * @brief Check if a Python package is installed in a given Python environment
     *
     * @param python_path Path to Python executable
     * @param package_name Name of package to check
     * @return true if package is installed, false otherwise
     */
    static bool hasPackage(const std::string& python_path, const std::string& package_name);

    /**
     * @brief Get human-readable description of environment type
     *
     * @param type Environment type
     * @return String description (e.g., "Conda", "Virtual Environment", "System Python")
     */
    static std::string typeToString(Type type);

private:
    /**
     * @brief Try to detect conda environment
     * @return Info struct if conda environment detected, std::nullopt otherwise
     */
    static std::optional<Info> detectConda();

    /**
     * @brief Try to detect virtual environment (venv/virtualenv)
     * @return Info struct if venv detected, std::nullopt otherwise
     */
    static std::optional<Info> detectVirtualEnv();

    /**
     * @brief Try to detect system Python installation
     * @return Info struct if system Python found, std::nullopt otherwise
     */
    static std::optional<Info> detectSystemPython();

    /**
     * @brief Find conda.sh activation script in common locations
     * @param conda_prefix Path to conda installation (from CONDA_PREFIX)
     * @return Path to conda.sh if found, empty string otherwise
     */
    static std::string findCondaActivationScript(const std::string& conda_prefix);

    /**
     * @brief Validate requirements against detected environment
     * @param info Environment info to validate
     * @param requirements Requirements to check
     */
    static void validateRequirements(Info& info, const Requirements& requirements);

    /**
     * @brief Execute shell command and capture output
     * @param command Command to execute
     * @return Command output (stdout)
     */
    static std::string executeCommand(const std::string& command);

    /**
     * @brief Get environment variable value
     * @param name Environment variable name
     * @return Value if set, std::nullopt otherwise
     */
    static std::optional<std::string> getEnvVar(const std::string& name);
};

} // namespace utils
} // namespace thesis_project
