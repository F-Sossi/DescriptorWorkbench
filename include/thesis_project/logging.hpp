#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

namespace thesis_project {
    namespace logging {
        
        enum class LogLevel {
            DEBUG,
            INFO,
            WARNING,
            ERROR
        };
        
        class Logger {
        private:
            static LogLevel current_level_;
            
            static std::string getCurrentTime() {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                return ss.str();
            }
            
            static std::string levelToString(LogLevel level) {
                switch (level) {
                    case LogLevel::DEBUG: return "DEBUG";
                    case LogLevel::INFO: return "INFO";
                    case LogLevel::WARNING: return "WARN";
                    case LogLevel::ERROR: return "ERROR";
                    default: return "UNKNOWN";
                }
            }
            
        public:
            static void setLevel(LogLevel level) {
                current_level_ = level;
            }
            
            static void log(LogLevel level, const std::string& message) {
                if (level >= current_level_) {
                    std::cout << "[" << getCurrentTime() << "] "
                              << "[" << levelToString(level) << "] "
                              << message << std::endl;
                }
            }
            
            static void debug(const std::string& message) {
                log(LogLevel::DEBUG, message);
            }
            
            static void info(const std::string& message) {
                log(LogLevel::INFO, message);
            }
            
            static void warning(const std::string& message) {
                log(LogLevel::WARNING, message);
            }
            
            static void error(const std::string& message) {
                log(LogLevel::ERROR, message);
            }
        };
        
        // Initialize static member (inline to avoid multiple definitions)
        inline LogLevel Logger::current_level_ = LogLevel::INFO;
        
        // Convenience macros
        #define LOG_DEBUG(msg) thesis_project::logging::Logger::debug(msg)
        #define LOG_INFO(msg) thesis_project::logging::Logger::info(msg)
        #define LOG_WARNING(msg) thesis_project::logging::Logger::warning(msg)
        #define LOG_ERROR(msg) thesis_project::logging::Logger::error(msg)
        
    } // namespace logging
} // namespace thesis_project