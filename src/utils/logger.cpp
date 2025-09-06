// Simple implementation - include the header directly with relative path
#include "../../include/thesis_project/logging.hpp"

namespace thesis_project {
namespace logging {
    // Initialize static member
    LogLevel Logger::current_level_ = LogLevel::INFO;
}
}
