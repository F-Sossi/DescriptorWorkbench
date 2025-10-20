#include "PythonEnvironment.hpp"
#include "thesis_project/logging.hpp"
#include <cstdlib>
#include <filesystem>
#include <array>
#include <memory>
#include <sstream>


namespace thesis_project::utils {

// Static method implementations

PythonEnvironment::Info PythonEnvironment::detect(const Requirements& requirements) {
    // Try detection methods in order of preference
    // 1. Active conda environment (most specific)
    if (auto conda_info = detectConda()) {
        validateRequirements(*conda_info, requirements);
        if (conda_info->is_valid()) {
            LOG_INFO("Detected conda environment: " + conda_info->environment_name);
            return *conda_info;
        }
    }

    // 2. Active virtual environment
    if (auto venv_info = detectVirtualEnv()) {
        validateRequirements(*venv_info, requirements);
        if (venv_info->is_valid()) {
            LOG_INFO("Detected virtual environment: " + venv_info->environment_path);
            return *venv_info;
        }
    }

    // 3. System Python
    if (auto system_info = detectSystemPython()) {
        validateRequirements(*system_info, requirements);
        if (system_info->is_valid()) {
            LOG_INFO("Using system Python: " + system_info->python_path);
            return *system_info;
        } else {
            LOG_WARNING("System Python found but missing required packages");
            return *system_info; // Return anyway with missing_packages list
        }
    }

    // No valid Python environment found
    LOG_ERROR("No valid Python environment detected");
    return Info{}; // Returns UNKNOWN type
}

std::string PythonEnvironment::generateCommand(
    const std::string& script_path,
    const std::vector<std::string>& args,
    const std::optional<Info>& provided_info
) {
    // Detect environment if not provided
    const Info info = provided_info.value_or(detect({}));

    if (!info.is_valid()) {
        LOG_ERROR("Cannot generate Python command: no valid environment detected");
        return "";
    }

    std::ostringstream cmd;

    // Build command based on environment type
    switch (info.type) {
        case Type::CONDA:
            // Use bash with conda activation
            cmd << "/bin/bash -c \"";
            cmd << info.activation_prefix;
            cmd << " && " << info.python_path;
            break;

        case Type::VIRTUALENV:
            // Use bash with virtualenv activation
            cmd << "/bin/bash -c \"";
            cmd << info.activation_prefix;
            cmd << " && " << info.python_path;
            break;

        case Type::SYSTEM:
            // Direct Python execution (no activation needed)
            cmd << info.python_path;
            break;

        default:
            LOG_ERROR("Unknown Python environment type");
            return "";
    }

    // Add script path (properly quoted)
    cmd << " \"" << script_path << "\"";

    // Add arguments
    for (const auto& arg : args) {
        cmd << " " << arg;
    }

    // Close the bash -c string if needed
    if (info.type == Type::CONDA || info.type == Type::VIRTUALENV) {
        cmd << "\"";
    }

    return cmd.str();
}

bool PythonEnvironment::hasPackage(const std::string& python_path, const std::string& package_name) {
    // Try to import the package
    const std::string check_cmd = python_path + " -c \"import " + package_name + "\" 2>/dev/null";
    const int result = std::system(check_cmd.c_str());
    return (result == 0);
}

std::string PythonEnvironment::typeToString(Type type) {
    switch (type) {
        case Type::CONDA:       return "Conda";
        case Type::VIRTUALENV:  return "Virtual Environment";
        case Type::SYSTEM:      return "System Python";
        case Type::UNKNOWN:     return "Unknown";
        default:                return "Invalid";
    }
}

// Private helper methods

std::optional<PythonEnvironment::Info> PythonEnvironment::detectConda() {
    // Check if we're in a conda environment
    auto conda_prefix = getEnvVar("CONDA_PREFIX");
    auto conda_default_env = getEnvVar("CONDA_DEFAULT_ENV");

    if (!conda_prefix) {
        return std::nullopt;
    }

    Info info;
    info.type = Type::CONDA;
    info.environment_path = *conda_prefix;
    info.environment_name = conda_default_env.value_or("base");

    // Find Python executable in conda environment
    std::filesystem::path python_bin = std::filesystem::path(*conda_prefix) / "bin" / "python3";
    if (!std::filesystem::exists(python_bin)) {
        python_bin = std::filesystem::path(*conda_prefix) / "bin" / "python";
    }

    if (!std::filesystem::exists(python_bin)) {
        LOG_WARNING("Conda environment detected but Python executable not found");
        return std::nullopt;
    }

    info.python_path = python_bin.string();

    // Find conda activation script
    // Try multiple common locations for conda.sh
    std::vector<std::string> conda_roots;

    // 1. Parent of CONDA_PREFIX (e.g., /path/to/miniforge3/envs/myenv -> /path/to/miniforge3)
    std::filesystem::path prefix_path(*conda_prefix);
    if (prefix_path.parent_path().filename() == "envs") {
        conda_roots.push_back(prefix_path.parent_path().parent_path().string());
    } else {
        // Might be base environment
        conda_roots.push_back(*conda_prefix);
    }

    // 2. Check CONDA_EXE if available (points to conda binary)
    if (auto conda_exe = getEnvVar("CONDA_EXE")) {
        std::filesystem::path exe_path(*conda_exe);
        // conda executable is typically in <conda_root>/bin/conda or <conda_root>/condabin/conda
        if (exe_path.parent_path().filename() == "bin" || exe_path.parent_path().filename() == "condabin") {
            conda_roots.push_back(exe_path.parent_path().parent_path().string());
        }
    }

    // 3. Common conda installation paths
    conda_roots.emplace_back("/opt/conda");
    conda_roots.emplace_back("/usr/local/conda");
    if (auto home = getEnvVar("HOME")) {
        conda_roots.push_back(*home + "/miniconda3");
        conda_roots.push_back(*home + "/miniforge3");
        conda_roots.push_back(*home + "/anaconda3");
    }

    // Search for conda.sh
    for (const auto& root : conda_roots) {
        std::string conda_sh = findCondaActivationScript(root);
        if (!conda_sh.empty()) {
            // Build activation command
            info.activation_prefix = "source " + conda_sh + " && conda activate " + info.environment_name;
            LOG_INFO("Found conda activation script: " + conda_sh);
            return info;
        }
    }

    // If we can't find conda.sh, we can still use the Python executable directly
    // This works in Docker or when conda is already activated
    LOG_WARNING("Could not find conda.sh, will use Python directly from: " + info.python_path);
    info.activation_prefix = ""; // No activation needed
    return info;
}

std::optional<PythonEnvironment::Info> PythonEnvironment::detectVirtualEnv() {
    const auto venv_path = getEnvVar("VIRTUAL_ENV");
    if (!venv_path) {
        return std::nullopt;
    }

    Info info;
    info.type = Type::VIRTUALENV;
    info.environment_path = *venv_path;
    info.environment_name = std::filesystem::path(*venv_path).filename().string();

    // Find Python executable
    std::filesystem::path python_bin = std::filesystem::path(*venv_path) / "bin" / "python3";
    if (!std::filesystem::exists(python_bin)) {
        python_bin = std::filesystem::path(*venv_path) / "bin" / "python";
    }

    if (!std::filesystem::exists(python_bin)) {
        LOG_WARNING("Virtual environment detected but Python executable not found");
        return std::nullopt;
    }

    info.python_path = python_bin.string();

    // Build activation command
    std::filesystem::path activate_script = std::filesystem::path(*venv_path) / "bin" / "activate";
    if (std::filesystem::exists(activate_script)) {
        info.activation_prefix = "source " + activate_script.string();
    } else {
        // Activation script not found, but we can still use Python directly
        info.activation_prefix = "";
    }

    return info;
}

std::optional<PythonEnvironment::Info> PythonEnvironment::detectSystemPython() {
    // Try to find Python in PATH
    std::string which_output = executeCommand("which python3 2>/dev/null");
    if (which_output.empty()) {
        which_output = executeCommand("which python 2>/dev/null");
    }

    if (which_output.empty()) {
        return std::nullopt;
    }

    // Remove trailing newline
    while (!which_output.empty() && (which_output.back() == '\n' || which_output.back() == '\r')) {
        which_output.pop_back();
    }

    Info info;
    info.type = Type::SYSTEM;
    info.python_path = which_output;
    info.environment_name = "system";
    info.environment_path = std::filesystem::path(which_output).parent_path().parent_path().string();
    info.activation_prefix = ""; // No activation needed for system Python

    return info;
}

std::string PythonEnvironment::findCondaActivationScript(const std::string& conda_prefix) {
    std::vector<std::string> possible_paths = {
        conda_prefix + "/etc/profile.d/conda.sh",
        conda_prefix + "/etc/conda/activate.d/conda.sh",
        conda_prefix + "/Scripts/conda.sh",  // Windows/MinGW
    };

    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return "";
}

void PythonEnvironment::validateRequirements(Info& info, const Requirements& requirements) {
    info.missing_packages.clear();

    // Check each required package
    for (const auto& package : requirements.packages) {
        if (!hasPackage(info.python_path, package)) {
            info.missing_packages.push_back(package);
        }
    }

    info.has_required_packages = info.missing_packages.empty();

    // Log missing packages
    if (!info.has_required_packages) {
        std::ostringstream msg;
        msg << "Missing Python packages in " << typeToString(info.type) << ": ";
        for (size_t i = 0; i < info.missing_packages.size(); ++i) {
            if (i > 0) msg << ", ";
            msg << info.missing_packages[i];
        }
        LOG_WARNING(msg.str());
    }
}

std::string PythonEnvironment::executeCommand(const std::string& command) {
    std::array<char, 128> buffer;
    std::string result;

    // Open pipe to command
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        return "";
    }

    // Read output
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    return result;
}

std::optional<std::string> PythonEnvironment::getEnvVar(const std::string& name) {
    const char* value = std::getenv(name.c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
}

} // namespace thesis_project::utils

