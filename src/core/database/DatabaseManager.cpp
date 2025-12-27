#include "thesis_project/database/DatabaseManager.hpp"
#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <tuple>
#include <map>
#include <unordered_map>

namespace thesis_project::database {

// PIMPL implementation to hide SQLite details
class DatabaseManager::Impl {
public:
    sqlite3* db = nullptr;
    DatabaseConfig config;
    bool enabled = false;

    explicit Impl(const DatabaseConfig& cfg) : config(cfg), enabled(cfg.enabled) {
        if (!enabled) {
            std::cout << "DatabaseManager: Disabled - no experiment tracking" << std::endl;
            return;
        }

        // Try to open database
        int rc = sqlite3_open(config.connection_string.c_str(), &db);
        if (rc != SQLITE_OK) {
            std::cerr << "DatabaseManager: Failed to open database: "
                      << sqlite3_errmsg(db) << std::endl;
            enabled = false;
            if (db) {
                sqlite3_close(db);
                db = nullptr;
            }
        } else {
            std::cout << "DatabaseManager: Connected to " << config.connection_string << std::endl;
        }
    }

    ~Impl() {
        if (db) {
            sqlite3_close(db);
        }
    }

    bool initializeTables() const {
        if (!enabled || !db) return !enabled; // Success if disabled

        const auto create_experiments_table = R"(
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                descriptor_type TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                pooling_strategy TEXT,
                similarity_threshold REAL,
                max_features INTEGER,
                timestamp TEXT NOT NULL,
                parameters TEXT,
                keypoint_set_id INTEGER DEFAULT NULL,
                keypoint_source TEXT DEFAULT NULL,
                descriptor_dimension INTEGER DEFAULT 0,
                execution_device TEXT DEFAULT 'cpu',
                FOREIGN KEY(keypoint_set_id) REFERENCES keypoint_sets(id)
            );
        )";

        const auto create_results_table = R"(
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                -- PRIMARY IR-style mAP metrics (v2.0 schema upgrade)
                true_map_macro REAL,                    -- Scene-balanced mAP (primary metric)
                true_map_micro REAL,                    -- Overall mAP weighted by query count
                true_map_macro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0
                true_map_micro_with_zeros REAL,         -- Conservative: includes R=0 queries as AP=0
                image_retrieval_map REAL DEFAULT -1,    -- Image-level retrieval MAP (optional)
                -- Category-specific metrics (v3.1): Viewpoint vs Illumination
                viewpoint_map REAL DEFAULT 0.0,         -- mAP for v_* sequences only (geometric changes)
                illumination_map REAL DEFAULT 0.0,      -- mAP for i_* sequences only (photometric changes)
                viewpoint_map_with_zeros REAL DEFAULT 0.0,     -- Conservative: includes R=0 queries
                illumination_map_with_zeros REAL DEFAULT 0.0,  -- Conservative: includes R=0 queries
                -- Keypoint verification metrics (v3.2): Bojanic et al. (2020) verification task
                keypoint_verification_ap REAL DEFAULT -1.0,    -- Verification AP with distractors (-1 when disabled)
                verification_viewpoint_ap REAL DEFAULT -1.0,   -- Verification AP for viewpoint scenes only
                verification_illumination_ap REAL DEFAULT -1.0, -- Verification AP for illumination scenes only
                -- Keypoint retrieval metrics (v3.3): Bojanic et al. (2020) retrieval task
                keypoint_retrieval_ap REAL DEFAULT -1.0,       -- Retrieval AP with three-tier labels (-1 when disabled)
                retrieval_viewpoint_ap REAL DEFAULT -1.0,      -- Retrieval AP for viewpoint scenes only
                retrieval_illumination_ap REAL DEFAULT -1.0,   -- Retrieval AP for illumination scenes only
                retrieval_num_true_positives INTEGER DEFAULT 0, -- Count of y=+1 labels
                retrieval_num_hard_negatives INTEGER DEFAULT 0, -- Count of y=0 labels
                retrieval_num_distractors INTEGER DEFAULT 0,   -- Count of y=-1 labels
                -- Legacy/compatibility metrics
                mean_average_precision REAL,            -- Primary display metric (uses true_map_macro when available)
                legacy_mean_precision REAL,             -- Original arithmetic mean for backward compatibility
                -- Standard retrieval metrics
                precision_at_1 REAL,
                precision_at_5 REAL,
                recall_at_1 REAL,
                recall_at_5 REAL,
                -- Experiment metadata
                total_matches INTEGER,
                total_keypoints INTEGER,
                processing_time_ms REAL,
                timestamp TEXT NOT NULL,
                descriptor_time_cpu_ms REAL,
                descriptor_time_gpu_ms REAL,
                match_time_cpu_ms REAL,
                match_time_gpu_ms REAL,
                total_pipeline_cpu_ms REAL,
                total_pipeline_gpu_ms REAL,
                metadata TEXT,                          -- Additional metrics and profiling data
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
        )";

        const auto create_keypoint_sets_table = R"(
            CREATE TABLE IF NOT EXISTS keypoint_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                generator_type TEXT NOT NULL,
                generation_method TEXT NOT NULL,
                max_features INTEGER,
                dataset_path TEXT,
                description TEXT,
                boundary_filter_px INTEGER DEFAULT 40,
                overlap_filtering BOOLEAN DEFAULT FALSE,
                min_distance REAL DEFAULT 0.0,
                source_set_a_id INTEGER DEFAULT NULL,
                source_set_b_id INTEGER DEFAULT NULL,
                tolerance_px REAL DEFAULT NULL,
                intersection_method TEXT DEFAULT NULL,
                detection_time_cpu_ms REAL,
                detection_time_gpu_ms REAL,
                total_generation_cpu_ms REAL,
                total_generation_gpu_ms REAL,
                intersection_time_ms REAL,
                avg_keypoints_per_image REAL,
                total_keypoints INTEGER,
                source_a_keypoints INTEGER,
                source_b_keypoints INTEGER,
                keypoint_reduction_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_set_a_id) REFERENCES keypoint_sets(id),
                FOREIGN KEY(source_set_b_id) REFERENCES keypoint_sets(id)
            );
        )";

        const auto create_keypoints_table = R"(
            CREATE TABLE IF NOT EXISTS locked_keypoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keypoint_set_id INTEGER NOT NULL DEFAULT 1,
                scene_name TEXT NOT NULL,
                image_name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                size REAL NOT NULL,
                angle REAL NOT NULL,
                response REAL NOT NULL,
                octave INTEGER NOT NULL,
                class_id INTEGER NOT NULL,
                valid_bounds BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(keypoint_set_id) REFERENCES keypoint_sets(id),
                UNIQUE(keypoint_set_id, scene_name, image_name, x, y, size, angle, response, octave)
            );
        )";

        const auto create_detector_attributes_table = R"(
            CREATE TABLE IF NOT EXISTS keypoint_detector_attributes (
                locked_keypoint_id INTEGER NOT NULL,
                detector_type TEXT NOT NULL,
                size REAL,
                angle REAL,
                response REAL,
                octave INTEGER,
                class_id INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (locked_keypoint_id, detector_type),
                FOREIGN KEY(locked_keypoint_id) REFERENCES locked_keypoints(id)
            );
        )";

        const auto create_descriptors_table = R"(
            CREATE TABLE IF NOT EXISTS descriptors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                scene_name TEXT NOT NULL,
                image_name TEXT NOT NULL,
                keypoint_x REAL NOT NULL,
                keypoint_y REAL NOT NULL,
                descriptor_vector BLOB NOT NULL,
                descriptor_dimension INTEGER NOT NULL,
                processing_method TEXT,
                normalization_applied TEXT,
                rooting_applied TEXT,
                pooling_applied TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                UNIQUE(experiment_id, scene_name, image_name, keypoint_x, keypoint_y)
            );
        )";

        const auto create_matches_table = R"(
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                scene_name TEXT NOT NULL,
                query_image TEXT NOT NULL,
                train_image TEXT NOT NULL,
                query_keypoint_x REAL NOT NULL,
                query_keypoint_y REAL NOT NULL,
                train_keypoint_x REAL NOT NULL,
                train_keypoint_y REAL NOT NULL,
                distance REAL NOT NULL,
                match_confidence REAL,
                is_correct_match BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
        )";

        const auto create_visualizations_table = R"(
            CREATE TABLE IF NOT EXISTS visualizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                scene_name TEXT NOT NULL,
                visualization_type TEXT NOT NULL,
                image_pair TEXT,
                image_data BLOB NOT NULL,
                image_format TEXT DEFAULT 'PNG',
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            );
        )";

        const auto create_keypoint_indexes = R"(
            CREATE INDEX IF NOT EXISTS idx_keypoint_sets_method ON keypoint_sets(generation_method);
            CREATE INDEX IF NOT EXISTS idx_keypoint_sets_generator ON keypoint_sets(generator_type);
            CREATE INDEX IF NOT EXISTS idx_keypoint_sets_overlap ON keypoint_sets(overlap_filtering);
            CREATE INDEX IF NOT EXISTS idx_locked_keypoints_set ON locked_keypoints(keypoint_set_id);
            CREATE INDEX IF NOT EXISTS idx_locked_keypoints_scene ON locked_keypoints(keypoint_set_id, scene_name, image_name);
            CREATE INDEX IF NOT EXISTS idx_kp_attr_detector ON keypoint_detector_attributes(detector_type);
            CREATE INDEX IF NOT EXISTS idx_kp_attr_locked_id ON keypoint_detector_attributes(locked_keypoint_id);
        )";

        const auto create_descriptor_indexes = R"(
            CREATE INDEX IF NOT EXISTS idx_descriptors_experiment ON descriptors(experiment_id, processing_method);
            CREATE INDEX IF NOT EXISTS idx_descriptors_keypoint ON descriptors(scene_name, image_name, keypoint_x, keypoint_y);
            CREATE INDEX IF NOT EXISTS idx_descriptors_method ON descriptors(processing_method, normalization_applied, rooting_applied);
        )";

        const auto create_matches_indexes = R"(
            CREATE INDEX IF NOT EXISTS idx_matches_experiment ON matches(experiment_id, scene_name);
            CREATE INDEX IF NOT EXISTS idx_matches_correctness ON matches(experiment_id, is_correct_match);
            CREATE INDEX IF NOT EXISTS idx_matches_image_pair ON matches(experiment_id, scene_name, query_image, train_image);
        )";

        const auto create_visualizations_indexes = R"(
            CREATE INDEX IF NOT EXISTS idx_visualizations_experiment ON visualizations(experiment_id, scene_name);
            CREATE INDEX IF NOT EXISTS idx_visualizations_type ON visualizations(visualization_type);
            CREATE INDEX IF NOT EXISTS idx_visualizations_pair ON visualizations(experiment_id, scene_name, image_pair);
        )";

        char* error_msg = nullptr;

        int rc1 = sqlite3_exec(db, create_experiments_table, nullptr, nullptr, &error_msg);
        if (rc1 != SQLITE_OK) {
            std::cerr << "Failed to create experiments table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc2 = sqlite3_exec(db, create_results_table, nullptr, nullptr, &error_msg);
        if (rc2 != SQLITE_OK) {
            std::cerr << "Failed to create results table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        auto ensure_results_column = [&](const std::string& column_name,
                                         const std::string& alter_statement) -> bool {
            sqlite3_stmt* stmt = nullptr;
            const std::string pragma = "PRAGMA table_info(results);";
            bool found = false;

            if (sqlite3_prepare_v2(db, pragma.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
                std::cerr << "Failed to inspect results table schema: "
                          << sqlite3_errmsg(db) << std::endl;
                return false;
            }

            while (sqlite3_step(stmt) == SQLITE_ROW) {
                const unsigned char* text = sqlite3_column_text(stmt, 1);
                if (text && column_name == reinterpret_cast<const char*>(text)) {
                    found = true;
                    break;
                }
            }
            sqlite3_finalize(stmt);

            if (found) {
                return true;
            }

            char* alter_err = nullptr;
            if (sqlite3_exec(db, alter_statement.c_str(), nullptr, nullptr, &alter_err) != SQLITE_OK) {
                std::cerr << "Failed to add column '" << column_name
                          << "' to results table: " << alter_err << std::endl;
                sqlite3_free(alter_err);
                return false;
            }

            std::cout << "Database upgrade: added results." << column_name << std::endl;
            return true;
        };

        // Ensure category split columns exist (v3.1)
        if (!ensure_results_column("viewpoint_map", "ALTER TABLE results ADD COLUMN viewpoint_map REAL DEFAULT 0.0;")) return false;
        if (!ensure_results_column("illumination_map", "ALTER TABLE results ADD COLUMN illumination_map REAL DEFAULT 0.0;")) return false;
        if (!ensure_results_column("viewpoint_map_with_zeros", "ALTER TABLE results ADD COLUMN viewpoint_map_with_zeros REAL DEFAULT 0.0;")) return false;
        if (!ensure_results_column("illumination_map_with_zeros", "ALTER TABLE results ADD COLUMN illumination_map_with_zeros REAL DEFAULT 0.0;")) return false;

        // Ensure verification columns exist (v3.2)
        if (!ensure_results_column("keypoint_verification_ap", "ALTER TABLE results ADD COLUMN keypoint_verification_ap REAL DEFAULT -1.0;")) return false;
        if (!ensure_results_column("verification_viewpoint_ap", "ALTER TABLE results ADD COLUMN verification_viewpoint_ap REAL DEFAULT -1.0;")) return false;
        if (!ensure_results_column("verification_illumination_ap", "ALTER TABLE results ADD COLUMN verification_illumination_ap REAL DEFAULT -1.0;")) return false;

        // Ensure retrieval columns exist (v3.3)
        if (!ensure_results_column("keypoint_retrieval_ap", "ALTER TABLE results ADD COLUMN keypoint_retrieval_ap REAL DEFAULT -1.0;")) return false;
        if (!ensure_results_column("retrieval_viewpoint_ap", "ALTER TABLE results ADD COLUMN retrieval_viewpoint_ap REAL DEFAULT -1.0;")) return false;
        if (!ensure_results_column("retrieval_illumination_ap", "ALTER TABLE results ADD COLUMN retrieval_illumination_ap REAL DEFAULT -1.0;")) return false;
        if (!ensure_results_column("retrieval_num_true_positives", "ALTER TABLE results ADD COLUMN retrieval_num_true_positives INTEGER DEFAULT 0;")) return false;
        if (!ensure_results_column("retrieval_num_hard_negatives", "ALTER TABLE results ADD COLUMN retrieval_num_hard_negatives INTEGER DEFAULT 0;")) return false;
        if (!ensure_results_column("retrieval_num_distractors", "ALTER TABLE results ADD COLUMN retrieval_num_distractors INTEGER DEFAULT 0;")) return false;

        int rc3 = sqlite3_exec(db, create_keypoint_sets_table, nullptr, nullptr, &error_msg);
        if (rc3 != SQLITE_OK) {
            std::cerr << "Failed to create keypoint_sets table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc4 = sqlite3_exec(db, create_keypoints_table, nullptr, nullptr, &error_msg);
        if (rc4 != SQLITE_OK) {
            std::cerr << "Failed to create locked_keypoints table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc4b = sqlite3_exec(db, create_detector_attributes_table, nullptr, nullptr, &error_msg);
        if (rc4b != SQLITE_OK) {
            std::cerr << "Failed to create keypoint_detector_attributes table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc5 = sqlite3_exec(db, create_descriptors_table, nullptr, nullptr, &error_msg);
        if (rc5 != SQLITE_OK) {
            std::cerr << "Failed to create descriptors table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc6 = sqlite3_exec(db, create_keypoint_indexes, nullptr, nullptr, &error_msg);
        if (rc6 != SQLITE_OK) {
            std::cerr << "Failed to create keypoint indexes: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc7 = sqlite3_exec(db, create_descriptor_indexes, nullptr, nullptr, &error_msg);
        if (rc7 != SQLITE_OK) {
            std::cerr << "Failed to create descriptor indexes: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc8 = sqlite3_exec(db, create_matches_table, nullptr, nullptr, &error_msg);
        if (rc8 != SQLITE_OK) {
            std::cerr << "Failed to create matches table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc9 = sqlite3_exec(db, create_visualizations_table, nullptr, nullptr, &error_msg);
        if (rc9 != SQLITE_OK) {
            std::cerr << "Failed to create visualizations table: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc10 = sqlite3_exec(db, create_matches_indexes, nullptr, nullptr, &error_msg);
        if (rc10 != SQLITE_OK) {
            std::cerr << "Failed to create matches indexes: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        int rc11 = sqlite3_exec(db, create_visualizations_indexes, nullptr, nullptr, &error_msg);
        if (rc11 != SQLITE_OK) {
            std::cerr << "Failed to create visualizations indexes: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }

        return true;
    }

    static std::string getCurrentTimestamp() {
        const auto now = std::chrono::system_clock::now();
        const auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

// DatabaseManager implementation
DatabaseManager::DatabaseManager(const DatabaseConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
    if (impl_->enabled && !initializeTables()) {
        std::cerr << "DatabaseManager: Failed to initialize tables" << std::endl;
    }
}

DatabaseManager::DatabaseManager(const std::string& db_path, bool enabled)
    : DatabaseManager(enabled ? DatabaseConfig::sqlite(db_path) : DatabaseConfig::disabled()) {
}

DatabaseManager::~DatabaseManager() = default;

bool DatabaseManager::isEnabled() const {
    return impl_->enabled && impl_->db != nullptr;
}

bool DatabaseManager::optimizeForBulkOperations() const {
    if (!isEnabled()) return true; // Success if disabled

    char* error_msg = nullptr;
    
    // SQLite performance optimizations for bulk operations
    const char* optimizations[] = {
        "PRAGMA journal_mode = WAL;",        // Write-Ahead Logging for better concurrency
        "PRAGMA synchronous = NORMAL;",      // Faster than FULL, still safe
        "PRAGMA cache_size = 10000;",        // Increase cache size (10MB)
        "PRAGMA temp_store = MEMORY;",       // Store temp data in memory
        "PRAGMA mmap_size = 268435456;",     // Use memory mapping (256MB)
        "PRAGMA optimize;"                   // Optimize query planner
    };

    for (const char* pragma : optimizations) {
        int rc = sqlite3_exec(impl_->db, pragma, nullptr, nullptr, &error_msg);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to apply optimization: " << pragma 
                      << " Error: " << error_msg << std::endl;
            sqlite3_free(error_msg);
            return false;
        }
    }

    std::cout << "Database optimized for bulk operations" << std::endl;
    return true;
}

bool DatabaseManager::initializeTables() const {
    return impl_->initializeTables();
}

int DatabaseManager::recordConfiguration(const ExperimentConfig& config) const {
    if (!isEnabled()) return -1;

    const auto sql = R"(
        INSERT INTO experiments (descriptor_type, dataset_name, pooling_strategy,
                               similarity_threshold, max_features, timestamp, parameters,
                               keypoint_set_id, keypoint_source, descriptor_dimension, execution_device)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    // Build parameters string
    std::stringstream params_ss;
    for (const auto& [key, value] : config.parameters) {
        params_ss << key << "=" << value << ";";
    }
    std::string params_str = params_ss.str();

    // Bind parameters (11 total now)
    sqlite3_bind_text(stmt, 1, config.descriptor_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, config.dataset_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, config.pooling_strategy.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 4, config.similarity_threshold);
    sqlite3_bind_int(stmt, 5, config.max_features);
    sqlite3_bind_text(stmt, 6, impl_->getCurrentTimestamp().c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, params_str.c_str(), -1, SQLITE_STATIC);

    // Bind keypoint tracking fields (NULL if not set)
    if (config.keypoint_set_id > 0) {
        sqlite3_bind_int(stmt, 8, config.keypoint_set_id);
    } else {
        sqlite3_bind_null(stmt, 8);
    }

    if (!config.keypoint_source.empty()) {
        sqlite3_bind_text(stmt, 9, config.keypoint_source.c_str(), -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_null(stmt, 9);
    }
    sqlite3_bind_int(stmt, 10, config.descriptor_dimension);
    if (!config.execution_device.empty()) {
        sqlite3_bind_text(stmt, 11, config.execution_device.c_str(), -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_text(stmt, 11, "cpu", -1, SQLITE_STATIC);
    }

    rc = sqlite3_step(stmt);
    int experiment_id = -1;

    if (rc == SQLITE_DONE) {
        experiment_id = static_cast<int>(sqlite3_last_insert_rowid(impl_->db));
        std::cout << "Recorded experiment config with ID: " << experiment_id << std::endl;
    } else {
        std::cerr << "Failed to insert experiment: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return experiment_id;
}

bool DatabaseManager::updateExperimentDescriptorMetadata(int experiment_id,
                                                         int descriptor_dimension,
                                                         const std::string& execution_device) const {
    if (!isEnabled()) return true;
    if (experiment_id < 0) {
        std::cerr << "Invalid experiment id for descriptor metadata update" << std::endl;
        return false;
    }

    const char* sql = R"(
        UPDATE experiments
        SET descriptor_dimension = ?, execution_device = ?
        WHERE id = ?
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare experiment metadata update: "
                  << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, descriptor_dimension);
    if (!execution_device.empty()) {
        sqlite3_bind_text(stmt, 2, execution_device.c_str(), -1, SQLITE_STATIC);
    } else {
        sqlite3_bind_text(stmt, 2, "cpu", -1, SQLITE_STATIC);
    }
    sqlite3_bind_int(stmt, 3, experiment_id);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to update experiment metadata: "
                  << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::recordExperiment(const ExperimentResults& results) const {
    if (!isEnabled()) return true; // Success if disabled

    const auto sql = R"(
        INSERT INTO results (experiment_id, true_map_macro, true_map_micro,
                           true_map_macro_with_zeros, true_map_micro_with_zeros,
                           image_retrieval_map,
                           viewpoint_map, illumination_map,
                           viewpoint_map_with_zeros, illumination_map_with_zeros,
                           keypoint_verification_ap, verification_viewpoint_ap, verification_illumination_ap,
                           keypoint_retrieval_ap, retrieval_viewpoint_ap, retrieval_illumination_ap,
                           retrieval_num_true_positives, retrieval_num_hard_negatives, retrieval_num_distractors,
                           precision_at_1, precision_at_5, recall_at_1, recall_at_5,
                           mean_average_precision, legacy_mean_precision,
                           total_matches, total_keypoints, processing_time_ms,
                           timestamp, descriptor_time_cpu_ms, descriptor_time_gpu_ms,
                           match_time_cpu_ms, match_time_gpu_ms,
                           total_pipeline_cpu_ms, total_pipeline_gpu_ms,
                           metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    // Build metadata string
    std::stringstream metadata_ss;
    for (const auto& [key, value] : results.metadata) {
        metadata_ss << key << "=" << value << ";";
    }
    const std::string metadata_str = metadata_ss.str();

    // Bind parameters (36 total now - added 6 for retrieval metrics)
    sqlite3_bind_int(stmt, 1, results.experiment_id);
    sqlite3_bind_double(stmt, 2, results.true_map_macro);
    sqlite3_bind_double(stmt, 3, results.true_map_micro);
    sqlite3_bind_double(stmt, 4, results.true_map_macro_with_zeros);
    sqlite3_bind_double(stmt, 5, results.true_map_micro_with_zeros);
    sqlite3_bind_double(stmt, 6, results.image_retrieval_map);
    sqlite3_bind_double(stmt, 7, results.viewpoint_map);
    sqlite3_bind_double(stmt, 8, results.illumination_map);
    sqlite3_bind_double(stmt, 9, results.viewpoint_map_with_zeros);
    sqlite3_bind_double(stmt, 10, results.illumination_map_with_zeros);
    sqlite3_bind_double(stmt, 11, results.keypoint_verification_ap);
    sqlite3_bind_double(stmt, 12, results.verification_viewpoint_ap);
    sqlite3_bind_double(stmt, 13, results.verification_illumination_ap);
    sqlite3_bind_double(stmt, 14, results.keypoint_retrieval_ap);
    sqlite3_bind_double(stmt, 15, results.retrieval_viewpoint_ap);
    sqlite3_bind_double(stmt, 16, results.retrieval_illumination_ap);
    sqlite3_bind_int(stmt, 17, results.retrieval_num_true_positives);
    sqlite3_bind_int(stmt, 18, results.retrieval_num_hard_negatives);
    sqlite3_bind_int(stmt, 19, results.retrieval_num_distractors);
    sqlite3_bind_double(stmt, 20, results.precision_at_1);
    sqlite3_bind_double(stmt, 21, results.precision_at_5);
    sqlite3_bind_double(stmt, 22, results.recall_at_1);
    sqlite3_bind_double(stmt, 23, results.recall_at_5);
    sqlite3_bind_double(stmt, 24, results.mean_average_precision);
    sqlite3_bind_double(stmt, 25, results.legacy_mean_precision);
    sqlite3_bind_int(stmt, 26, results.total_matches);
    sqlite3_bind_int(stmt, 27, results.total_keypoints);
    sqlite3_bind_double(stmt, 28, results.processing_time_ms);
    sqlite3_bind_text(stmt, 29, impl_->getCurrentTimestamp().c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 30, results.descriptor_time_cpu_ms);
    sqlite3_bind_double(stmt, 31, results.descriptor_time_gpu_ms);
    sqlite3_bind_double(stmt, 32, results.match_time_cpu_ms);
    sqlite3_bind_double(stmt, 33, results.match_time_gpu_ms);
    sqlite3_bind_double(stmt, 34, results.total_pipeline_cpu_ms);
    sqlite3_bind_double(stmt, 35, results.total_pipeline_gpu_ms);
    sqlite3_bind_text(stmt, 36, metadata_str.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);

    if (success) {
        std::cout << "Recorded experiment results (MAP: "
                  << results.mean_average_precision << ")" << std::endl;
    } else {
        std::cerr << "Failed to insert results: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

std::vector<ExperimentResults> DatabaseManager::getRecentResults(int limit) const {
    std::vector<ExperimentResults> results;
    if (!isEnabled()) return results;

    const auto sql = R"(
        SELECT r.experiment_id, e.descriptor_type, e.dataset_name,
               r.true_map_macro, r.true_map_micro,
               r.true_map_macro_with_zeros, r.true_map_micro_with_zeros,
               r.image_retrieval_map,
               r.viewpoint_map, r.illumination_map,
               r.viewpoint_map_with_zeros, r.illumination_map_with_zeros,
               r.keypoint_verification_ap, r.verification_viewpoint_ap, r.verification_illumination_ap,
               r.keypoint_retrieval_ap, r.retrieval_viewpoint_ap, r.retrieval_illumination_ap,
               r.retrieval_num_true_positives, r.retrieval_num_hard_negatives, r.retrieval_num_distractors,
               r.mean_average_precision, r.legacy_mean_precision,
               r.precision_at_1, r.precision_at_5,
               r.recall_at_1, r.recall_at_5,
               r.total_matches, r.total_keypoints,
               r.processing_time_ms, r.timestamp
        FROM results r
        JOIN experiments e ON r.experiment_id = e.id
        ORDER BY r.timestamp DESC
        LIMIT ?;
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return results;
    }

    sqlite3_bind_int(stmt, 1, limit);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        ExperimentResults result;
        result.experiment_id = sqlite3_column_int(stmt, 0);
        result.descriptor_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.dataset_name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.true_map_macro = sqlite3_column_double(stmt, 3);
        result.true_map_micro = sqlite3_column_double(stmt, 4);
        result.true_map_macro_with_zeros = sqlite3_column_double(stmt, 5);
        result.true_map_micro_with_zeros = sqlite3_column_double(stmt, 6);
        result.image_retrieval_map = sqlite3_column_double(stmt, 7);
        result.viewpoint_map = sqlite3_column_double(stmt, 8);
        result.illumination_map = sqlite3_column_double(stmt, 9);
        result.viewpoint_map_with_zeros = sqlite3_column_double(stmt, 10);
        result.illumination_map_with_zeros = sqlite3_column_double(stmt, 11);
        result.keypoint_verification_ap = sqlite3_column_double(stmt, 12);
        result.verification_viewpoint_ap = sqlite3_column_double(stmt, 13);
        result.verification_illumination_ap = sqlite3_column_double(stmt, 14);
        result.keypoint_retrieval_ap = sqlite3_column_double(stmt, 15);
        result.retrieval_viewpoint_ap = sqlite3_column_double(stmt, 16);
        result.retrieval_illumination_ap = sqlite3_column_double(stmt, 17);
        result.retrieval_num_true_positives = sqlite3_column_int(stmt, 18);
        result.retrieval_num_hard_negatives = sqlite3_column_int(stmt, 19);
        result.retrieval_num_distractors = sqlite3_column_int(stmt, 20);
        result.mean_average_precision = sqlite3_column_double(stmt, 21);
        result.legacy_mean_precision = sqlite3_column_double(stmt, 22);
        result.precision_at_1 = sqlite3_column_double(stmt, 23);
        result.precision_at_5 = sqlite3_column_double(stmt, 24);
        result.recall_at_1 = sqlite3_column_double(stmt, 25);
        result.recall_at_5 = sqlite3_column_double(stmt, 26);
        result.total_matches = sqlite3_column_int(stmt, 27);
        result.total_keypoints = sqlite3_column_int(stmt, 28);
        result.processing_time_ms = sqlite3_column_double(stmt, 29);
        result.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 30));

        results.push_back(result);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::map<std::string, double> DatabaseManager::getStatistics() const {
    std::map<std::string, double> stats;
    if (!isEnabled()) return stats;

    const char* sql = R"(
        SELECT
            COUNT(*) as total_experiments,
            AVG(mean_average_precision) as avg_map,
            MAX(mean_average_precision) as best_map,
            AVG(processing_time_ms) as avg_time
        FROM results;
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return stats;
    }

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        stats["total_experiments"] = sqlite3_column_double(stmt, 0);
        stats["average_map"] = sqlite3_column_double(stmt, 1);
        stats["best_map"] = sqlite3_column_double(stmt, 2);
        stats["average_time_ms"] = sqlite3_column_double(stmt, 3);
    }

    sqlite3_finalize(stmt);
    return stats;
}

bool DatabaseManager::storeLockedKeypoints(const std::string& scene_name, const std::string& image_name, const std::vector<cv::KeyPoint>& keypoints) const {
    if (!isEnabled()) return true; // Success if disabled

    if (keypoints.empty()) {
        std::cout << "No keypoints to store for " << scene_name << "/" << image_name << std::endl;
        return true;
    }

    // Start transaction for massive performance improvement
    sqlite3_exec(impl_->db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    // First, clear existing keypoints for this scene/image
    const auto clear_sql = "DELETE FROM locked_keypoints WHERE scene_name = ? AND image_name = ?";
    sqlite3_stmt* clear_stmt;
    int rc = sqlite3_prepare_v2(impl_->db, clear_sql, -1, &clear_stmt, nullptr);
    if (rc == SQLITE_OK) {
        sqlite3_bind_text(clear_stmt, 1, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(clear_stmt, 2, image_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(clear_stmt);
    }
    sqlite3_finalize(clear_stmt);

    // Use optimized batch insert with prepared statement
    const auto sql = R"(
        INSERT INTO locked_keypoints (scene_name, image_name, x, y, size, angle, response, octave, class_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    )";

    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
        return false;
    }

    size_t stored_count = 0;
    bool success = true;

    // Batch insert all keypoints within single transaction
    for (const auto& kp : keypoints) {
        sqlite3_bind_text(stmt, 1, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, image_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 3, kp.pt.x);
        sqlite3_bind_double(stmt, 4, kp.pt.y);
        sqlite3_bind_double(stmt, 5, kp.size);
        sqlite3_bind_double(stmt, 6, kp.angle);
        sqlite3_bind_double(stmt, 7, kp.response);
        sqlite3_bind_int(stmt, 8, kp.octave);
        sqlite3_bind_int(stmt, 9, kp.class_id);

        rc = sqlite3_step(stmt);
        if (rc == SQLITE_DONE) {
            stored_count++;
        } else {
            std::cerr << "Failed to insert keypoint: " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
            break;
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
        std::cout << "Stored " << stored_count << " keypoints for " << scene_name << "/" << image_name << std::endl;
    } else {
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
        std::cerr << "Failed to store keypoints for " << scene_name << "/" << image_name << std::endl;
    }
    
    return success && (stored_count == keypoints.size());
}

std::vector<cv::KeyPoint> DatabaseManager::getLockedKeypoints(const std::string& scene_name, const std::string& image_name) const {
    std::vector<cv::KeyPoint> keypoints;
    if (!isEnabled()) return keypoints;

    const char* sql = R"(
        SELECT x, y, size, angle, response, octave, class_id
        FROM locked_keypoints
        WHERE scene_name = ? AND image_name = ?
        ORDER BY id;
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return keypoints;
    }

    sqlite3_bind_text(stmt, 1, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, image_name.c_str(), -1, SQLITE_STATIC);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        cv::KeyPoint kp;
        kp.pt.x = sqlite3_column_double(stmt, 0);
        kp.pt.y = sqlite3_column_double(stmt, 1);
        kp.size = sqlite3_column_double(stmt, 2);
        kp.angle = sqlite3_column_double(stmt, 3);
        kp.response = sqlite3_column_double(stmt, 4);
        kp.octave = sqlite3_column_int(stmt, 5);
        kp.class_id = sqlite3_column_int(stmt, 6);
        keypoints.push_back(kp);
    }

    sqlite3_finalize(stmt);
    return keypoints;
}

std::vector<std::string> DatabaseManager::getAvailableScenes() const {
    std::vector<std::string> scenes;
    if (!isEnabled()) return scenes;

    const char* sql = "SELECT DISTINCT scene_name FROM locked_keypoints ORDER BY scene_name;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return scenes;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        scenes.push_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }

    sqlite3_finalize(stmt);
    return scenes;
}

std::vector<std::string> DatabaseManager::getAvailableImages(const std::string& scene_name) const {
    std::vector<std::string> images;
    if (!isEnabled()) return images;

    const char* sql = "SELECT DISTINCT image_name FROM locked_keypoints WHERE scene_name = ? ORDER BY image_name;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return images;
    }

    sqlite3_bind_text(stmt, 1, scene_name.c_str(), -1, SQLITE_STATIC);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        images.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
    }

    sqlite3_finalize(stmt);
    return images;
}

bool DatabaseManager::clearSceneKeypoints(const std::string& scene_name) const {
    if (!isEnabled()) return true; // Success if disabled

    const auto sql = "DELETE FROM locked_keypoints WHERE scene_name = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare clear statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, scene_name.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    
    bool success = (rc == SQLITE_DONE);
    if (success) {
        int deleted_count = sqlite3_changes(impl_->db);
        std::cout << "Cleared " << deleted_count << " keypoints for scene: " << scene_name << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

int DatabaseManager::getKeypointSetId(const std::string& set_name) const {
    if (!isEnabled()) return -1;

    const auto sql = "SELECT id FROM keypoint_sets WHERE name = ? LIMIT 1;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return -1;
    }

    sqlite3_bind_text(stmt, 1, set_name.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    int set_id = -1;
    if (rc == SQLITE_ROW) {
        set_id = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return set_id;
}

bool DatabaseManager::storeDescriptors(int experiment_id,
                                      const std::string& scene_name,
                                      const std::string& image_name,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      const cv::Mat& descriptors,
                                      const std::string& processing_method,
                                      const std::string& normalization_applied,
                                      const std::string& rooting_applied,
                                      const std::string& pooling_applied) const {
    if (!impl_->enabled || !impl_->db) return !impl_->enabled;

    if (keypoints.size() != static_cast<size_t>(descriptors.rows)) {
        std::cerr << "Error: Keypoints count (" << keypoints.size() 
                  << ") does not match descriptor rows (" << descriptors.rows << ")" << std::endl;
        return false;
    }

    const auto sql = R"(
        INSERT OR REPLACE INTO descriptors 
        (experiment_id, scene_name, image_name, keypoint_x, keypoint_y, 
         descriptor_vector, descriptor_dimension, processing_method, 
         normalization_applied, rooting_applied, pooling_applied) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare descriptor insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    // Begin transaction for efficiency
    sqlite3_exec(impl_->db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    bool success = true;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::KeyPoint& kp = keypoints[i];
        cv::Mat descriptor_row = descriptors.row(i);

        // Bind parameters
        sqlite3_bind_int(stmt, 1, experiment_id);
        sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, kp.pt.x);
        sqlite3_bind_double(stmt, 5, kp.pt.y);

        // Store descriptor as binary blob
        sqlite3_bind_blob(stmt, 6, descriptor_row.data, 
                         descriptor_row.total() * descriptor_row.elemSize(), SQLITE_STATIC);
        sqlite3_bind_int(stmt, 7, descriptor_row.cols);
        sqlite3_bind_text(stmt, 8, processing_method.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 9, normalization_applied.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 10, rooting_applied.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 11, pooling_applied.c_str(), -1, SQLITE_STATIC);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert descriptor " << i << ": " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
            break;
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
        std::cout << "Stored " << keypoints.size() << " descriptors for " 
                  << scene_name << "/" << image_name << " (experiment " << experiment_id << ")" << std::endl;
    } else {
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
    }

    return success;
}

cv::Mat DatabaseManager::getDescriptors(int experiment_id,
                                       const std::string& scene_name,
                                       const std::string& image_name) const {
    if (!impl_->enabled || !impl_->db) return cv::Mat();

    const char* sql = R"(
        SELECT descriptor_vector, descriptor_dimension 
        FROM descriptors 
        WHERE experiment_id = ? AND scene_name = ? AND image_name = ?
        ORDER BY keypoint_x, keypoint_y
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare descriptor select statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return {};
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);

    std::vector<cv::Mat> descriptor_rows;
    int descriptor_dim = 0;

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const void* blob_data = sqlite3_column_blob(stmt, 0);
        int blob_size = sqlite3_column_bytes(stmt, 0);
        descriptor_dim = sqlite3_column_int(stmt, 1);

        // Create cv::Mat from blob data
        cv::Mat row(1, descriptor_dim, CV_32F);
        memcpy(row.data, blob_data, blob_size);
        descriptor_rows.push_back(row);
    }

    sqlite3_finalize(stmt);

    if (descriptor_rows.empty()) {
        return {};
    }

    // Combine all rows into single Mat
    cv::Mat result;
    cv::vconcat(descriptor_rows, result);
    return result;
}

std::vector<std::tuple<std::string, std::string, cv::Mat>> DatabaseManager::getDescriptorsByMethod(
    const std::string& processing_method,
    const std::string& normalization_applied,
    const std::string& rooting_applied) const {
    
    std::vector<std::tuple<std::string, std::string, cv::Mat>> results;
    if (!impl_->enabled || !impl_->db) return results;

    std::string sql = "SELECT DISTINCT scene_name, image_name FROM descriptors WHERE processing_method = ?";
    std::vector<std::string> params = {processing_method};

    if (!normalization_applied.empty()) {
        sql += " AND normalization_applied = ?";
        params.push_back(normalization_applied);
    }
    if (!rooting_applied.empty()) {
        sql += " AND rooting_applied = ?";
        params.push_back(rooting_applied);
    }

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare descriptor method query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return results;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        sqlite3_bind_text(stmt, i + 1, params[i].c_str(), -1, SQLITE_STATIC);
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        std::string scene = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string image = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        
        // Get descriptors for this scene/image combination
        cv::Mat descriptors = getDescriptors(-1, scene, image); // Use -1 to get latest
        results.emplace_back(scene, image, descriptors);
    }

    sqlite3_finalize(stmt);
    return results;
}

std::vector<std::string> DatabaseManager::getAvailableProcessingMethods() const {
    std::vector<std::string> methods;
    if (!impl_->enabled || !impl_->db) return methods;

    const char* sql = "SELECT DISTINCT processing_method FROM descriptors ORDER BY processing_method";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare processing methods query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return methods;
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const auto method = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (method) {
            methods.emplace_back(method);
        }
    }

    sqlite3_finalize(stmt);
    return methods;
}

bool DatabaseManager::deleteDescriptorsForExperiment(int experiment_id) const {
    if (!impl_->enabled || !impl_->db) return !impl_->enabled;

    const char* sql = "DELETE FROM descriptors WHERE experiment_id = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare descriptor delete statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to delete descriptors for experiment " << experiment_id << ": "
                  << sqlite3_errmsg(impl_->db) << std::endl;
    } else {
        const int deleted = sqlite3_changes(impl_->db);
        std::cout << "Deleted " << deleted << " descriptors for experiment " << experiment_id << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::deleteMatchesForExperiment(int experiment_id) const {
    if (!impl_->enabled || !impl_->db) return !impl_->enabled;

    const char* sql = "DELETE FROM matches WHERE experiment_id = ?;";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare match delete statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to delete matches for experiment " << experiment_id << ": "
                  << sqlite3_errmsg(impl_->db) << std::endl;
    } else {
        const int deleted = sqlite3_changes(impl_->db);
        std::cout << "Deleted " << deleted << " matches for experiment " << experiment_id << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

int DatabaseManager::createKeypointSet(const std::string& name,
                                      const std::string& generator_type,
                                      const std::string& generation_method,
                                      int max_features,
                                      const std::string& dataset_path,
                                      const std::string& description,
                                      int boundary_filter_px) const {
    if (!impl_->enabled || !impl_->db) return -1;

    const auto sql = R"(
        INSERT INTO keypoint_sets (name, generator_type, generation_method, max_features, dataset_path, description, boundary_filter_px)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint set insert: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, generator_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, generation_method.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 4, max_features);
    sqlite3_bind_text(stmt, 5, dataset_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, description.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 7, boundary_filter_px);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert keypoint set: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    return static_cast<int>(sqlite3_last_insert_rowid(impl_->db));
}

int DatabaseManager::createKeypointSetWithOverlap(const std::string& name,
                                                 const std::string& generator_type,
                                                 const std::string& generation_method,
                                                 int max_features,
                                                 const std::string& dataset_path,
                                                 const std::string& description,
                                                 int boundary_filter_px,
                                                 bool overlap_filtering,
                                                 float min_distance) const {
    if (!impl_->enabled || !impl_->db) return -1;

    const auto sql = R"(
        INSERT INTO keypoint_sets (name, generator_type, generation_method, max_features, dataset_path, description, boundary_filter_px, overlap_filtering, min_distance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint set insert with overlap: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, generator_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, generation_method.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 4, max_features);
    sqlite3_bind_text(stmt, 5, dataset_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, description.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 7, boundary_filter_px);
    sqlite3_bind_int(stmt, 8, overlap_filtering ? 1 : 0);
    sqlite3_bind_double(stmt, 9, static_cast<double>(min_distance));

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert keypoint set with overlap: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    return static_cast<int>(sqlite3_last_insert_rowid(impl_->db));
}

int DatabaseManager::createIntersectionKeypointSet(const std::string& name,
                                                  const std::string& generator_type,
                                                  const std::string& generation_method,
                                                  int max_features,
                                                  const std::string& dataset_path,
                                                  const std::string& description,
                                                  int boundary_filter_px,
                                                  int source_set_a_id,
                                                  int source_set_b_id,
                                                  float tolerance_px,
                                                  const std::string& intersection_method) const {
    if (!impl_->enabled || !impl_->db) return -1;

    const auto sql = R"(
        INSERT INTO keypoint_sets (name, generator_type, generation_method, max_features, dataset_path, description, boundary_filter_px, source_set_a_id, source_set_b_id, tolerance_px, intersection_method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare intersection keypoint set statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, generator_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, generation_method.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 4, max_features);
    sqlite3_bind_text(stmt, 5, dataset_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, description.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 7, boundary_filter_px);
    sqlite3_bind_int(stmt, 8, source_set_a_id);
    sqlite3_bind_int(stmt, 9, source_set_b_id);
    sqlite3_bind_double(stmt, 10, static_cast<double>(tolerance_px));
    sqlite3_bind_text(stmt, 11, intersection_method.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert intersection keypoint set: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    return static_cast<int>(sqlite3_last_insert_rowid(impl_->db));
}

bool DatabaseManager::updateIntersectionKeypointSet(int keypoint_set_id,
                                                    const std::string& generator_type,
                                                    const std::string& generation_method,
                                                    int max_features,
                                                    const std::string& dataset_path,
                                                    const std::string& description,
                                                    int boundary_filter_px,
                                                    int source_set_a_id,
                                                    int source_set_b_id,
                                                    float tolerance_px,
                                                    const std::string& intersection_method) const {
    if (!impl_->enabled || !impl_->db) return true;

    const char* sql = R"(
        UPDATE keypoint_sets
        SET generator_type = ?,
            generation_method = ?,
            max_features = ?,
            dataset_path = ?,
            description = ?,
            boundary_filter_px = ?,
            source_set_a_id = ?,
            source_set_b_id = ?,
            tolerance_px = ?,
            intersection_method = ?
        WHERE id = ?
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare update intersection keypoint set statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, generator_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, generation_method.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, max_features);
    sqlite3_bind_text(stmt, 4, dataset_path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, description.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 6, boundary_filter_px);
    sqlite3_bind_int(stmt, 7, source_set_a_id);
    sqlite3_bind_int(stmt, 8, source_set_b_id);
    sqlite3_bind_double(stmt, 9, static_cast<double>(tolerance_px));
    sqlite3_bind_text(stmt, 10, intersection_method.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 11, keypoint_set_id);

    rc = sqlite3_step(stmt);
    bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to update intersection keypoint set: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::storeLockedKeypointsForSet(int keypoint_set_id, const std::string& scene_name,
                                                 const std::string& image_name, const std::vector<cv::KeyPoint>& keypoints) const {
    if (!impl_->enabled || !impl_->db) return true;

    const auto sql = R"(
        INSERT OR IGNORE INTO locked_keypoints 
        (keypoint_set_id, scene_name, image_name, x, y, size, angle, response, octave, class_id, valid_bounds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint insert: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    // Begin transaction for better performance
    sqlite3_exec(impl_->db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    bool success = true;
    for (const auto& kp : keypoints) {
        sqlite3_bind_int(stmt, 1, keypoint_set_id);
        sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, kp.pt.x);
        sqlite3_bind_double(stmt, 5, kp.pt.y);
        sqlite3_bind_double(stmt, 6, kp.size);
        sqlite3_bind_double(stmt, 7, kp.angle);
        sqlite3_bind_double(stmt, 8, kp.response);
        sqlite3_bind_int(stmt, 9, kp.octave);
        sqlite3_bind_int(stmt, 10, kp.class_id);
        sqlite3_bind_int(stmt, 11, 1); // valid_bounds = true

        int step_rc = sqlite3_step(stmt);
        if (step_rc != SQLITE_DONE) {
            std::cerr << "Failed to insert keypoint: " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
        }
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);
    sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
    return success;
}

std::vector<cv::KeyPoint> DatabaseManager::getLockedKeypointsFromSet(int keypoint_set_id, const std::string& scene_name,
                                                                     const std::string& image_name) const {
    std::vector<cv::KeyPoint> keypoints;
    if (!impl_->enabled || !impl_->db) return keypoints;

    const auto sql = R"(
        SELECT x, y, size, angle, response, octave, class_id
        FROM locked_keypoints
        WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
        ORDER BY response DESC
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return keypoints;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        cv::KeyPoint kp;
        kp.pt.x = sqlite3_column_double(stmt, 0);
        kp.pt.y = sqlite3_column_double(stmt, 1);
        kp.size = sqlite3_column_double(stmt, 2);
        kp.angle = sqlite3_column_double(stmt, 3);
        kp.response = sqlite3_column_double(stmt, 4);
        kp.octave = sqlite3_column_int(stmt, 5);
        kp.class_id = sqlite3_column_int(stmt, 6);
        keypoints.push_back(kp);
    }

    sqlite3_finalize(stmt);
    return keypoints;
}

std::vector<DatabaseManager::KeypointRecord> DatabaseManager::getLockedKeypointsWithIds(
    int keypoint_set_id,
    const std::string& scene_name,
    const std::string& image_name) const {

    std::vector<KeypointRecord> records;
    if (!impl_->enabled || !impl_->db) return records;

    const auto sql = R"(
        SELECT id, x, y, size, angle, response, octave, class_id
        FROM locked_keypoints
        WHERE keypoint_set_id = ? AND scene_name = ? AND image_name = ?
        ORDER BY response DESC
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint id query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return records;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        KeypointRecord record;
        record.id = sqlite3_column_int(stmt, 0);
        record.keypoint.pt.x = sqlite3_column_double(stmt, 1);
        record.keypoint.pt.y = sqlite3_column_double(stmt, 2);
        record.keypoint.size = sqlite3_column_double(stmt, 3);
        record.keypoint.angle = sqlite3_column_double(stmt, 4);
        record.keypoint.response = sqlite3_column_double(stmt, 5);
        record.keypoint.octave = sqlite3_column_int(stmt, 6);
        record.keypoint.class_id = sqlite3_column_int(stmt, 7);
        records.push_back(record);
    }

    sqlite3_finalize(stmt);
    return records;
}

int DatabaseManager::insertLockedKeypoint(int keypoint_set_id,
                                          const std::string& scene_name,
                                          const std::string& image_name,
                                          const cv::KeyPoint& keypoint,
                                          bool valid_bounds) const {
    if (!impl_->enabled || !impl_->db) return -1;

    const char* sql = R"(
        INSERT INTO locked_keypoints(
            keypoint_set_id, scene_name, image_name,
            x, y, size, angle, response, octave, class_id, valid_bounds)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare insertLockedKeypoint: " << sqlite3_errmsg(impl_->db) << std::endl;
        return -1;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 4, keypoint.pt.x);
    sqlite3_bind_double(stmt, 5, keypoint.pt.y);
    sqlite3_bind_double(stmt, 6, keypoint.size);
    sqlite3_bind_double(stmt, 7, keypoint.angle);
    sqlite3_bind_double(stmt, 8, keypoint.response);
    sqlite3_bind_int(stmt, 9, keypoint.octave);
    sqlite3_bind_int(stmt, 10, keypoint.class_id);
    sqlite3_bind_int(stmt, 11, valid_bounds ? 1 : 0);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert locked keypoint: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_finalize(stmt);
        return -1;
    }

    sqlite3_finalize(stmt);
    return static_cast<int>(sqlite3_last_insert_rowid(impl_->db));
}

int DatabaseManager::insertLockedKeypointsBatch(const int keypoint_set_id,
                                                 const std::string& scene_name,
                                                 const std::string& image_name,
                                                 const std::vector<cv::KeyPoint>& keypoints,
                                                 bool valid_bounds) const {
    if (!impl_->enabled || !impl_->db) return 0;
    if (keypoints.empty()) return 0;

    const auto sql = R"(
        INSERT INTO locked_keypoints(
            keypoint_set_id, scene_name, image_name,
            x, y, size, angle, response, octave, class_id, valid_bounds)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare batch insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return 0;
    }

    // Begin transaction for batched insert (Phase 1 optimization)
    rc = sqlite3_exec(impl_->db, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to begin transaction: " << sqlite3_errmsg(impl_->db) << std::endl;
        sqlite3_finalize(stmt);
        return 0;
    }

    int inserted_count = 0;
    bool success = true;

    for (const auto& keypoint : keypoints) {
        sqlite3_bind_int(stmt, 1, keypoint_set_id);
        sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, image_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 4, keypoint.pt.x);
        sqlite3_bind_double(stmt, 5, keypoint.pt.y);
        sqlite3_bind_double(stmt, 6, keypoint.size);
        sqlite3_bind_double(stmt, 7, keypoint.angle);
        sqlite3_bind_double(stmt, 8, keypoint.response);
        sqlite3_bind_int(stmt, 9, keypoint.octave);
        sqlite3_bind_int(stmt, 10, keypoint.class_id);
        sqlite3_bind_int(stmt, 11, valid_bounds ? 1 : 0);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert keypoint in batch: " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
            break;
        }

        ++inserted_count;
        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        rc = sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
        if (rc != SQLITE_OK) {
            std::cerr << "Failed to commit batch transaction: " << sqlite3_errmsg(impl_->db) << std::endl;
            sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
            return 0;
        }
    } else {
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
        return 0;
    }

    return inserted_count;
}

bool DatabaseManager::upsertDetectorAttributes(const std::vector<DetectorAttributeRecord>& records) const {
    if (!impl_->enabled || !impl_->db) return true;
    if (records.empty()) return true;

    const char* sql = R"(
        INSERT INTO keypoint_detector_attributes
        (locked_keypoint_id, detector_type, size, angle, response, octave, class_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(locked_keypoint_id, detector_type) DO UPDATE SET
            size = excluded.size,
            angle = excluded.angle,
            response = excluded.response,
            octave = excluded.octave,
            class_id = excluded.class_id,
            updated_at = CURRENT_TIMESTAMP
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare detector attribute upsert: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_exec(impl_->db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    bool success = true;
    for (const auto& record : records) {
        sqlite3_bind_int(stmt, 1, record.locked_keypoint_id);
        sqlite3_bind_text(stmt, 2, record.detector_type.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 3, record.attributes.size);
        sqlite3_bind_double(stmt, 4, record.attributes.angle);
        sqlite3_bind_double(stmt, 5, record.attributes.response);
        sqlite3_bind_int(stmt, 6, record.attributes.octave);
        sqlite3_bind_int(stmt, 7, record.attributes.class_id);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to upsert detector attribute: " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
            break;
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
    } else {
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
    }

    return success;
}

bool DatabaseManager::clearDetectorAttributesForSet(int keypoint_set_id, const std::string& detector_type) const {
    if (!impl_->enabled || !impl_->db) return true;

    const char* sql = R"(
        DELETE FROM keypoint_detector_attributes
        WHERE detector_type = ?
          AND locked_keypoint_id IN (
              SELECT id FROM locked_keypoints WHERE keypoint_set_id = ?
          )
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare detector attribute clear: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, detector_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, keypoint_set_id);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to clear detector attributes: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

std::unordered_map<int, DatabaseManager::DetectorAttributes> DatabaseManager::getDetectorAttributesForImage(
    const int keypoint_set_id,
    const std::string& scene_name,
    const std::string& image_name,
    const std::string& detector_type) const {

    std::unordered_map<int, DetectorAttributes> attributes_map;
    if (!impl_->enabled || !impl_->db) return attributes_map;

    const auto sql = R"(
        SELECT attr.locked_keypoint_id,
               attr.size,
               attr.angle,
               attr.response,
               attr.octave,
               attr.class_id
        FROM keypoint_detector_attributes attr
        INNER JOIN locked_keypoints lk ON lk.id = attr.locked_keypoint_id
        WHERE attr.detector_type = ?
          AND lk.keypoint_set_id = ?
          AND lk.scene_name = ?
          AND lk.image_name = ?
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare detector attribute query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return attributes_map;
    }

    sqlite3_bind_text(stmt, 1, detector_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, keypoint_set_id);
    sqlite3_bind_text(stmt, 3, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, image_name.c_str(), -1, SQLITE_STATIC);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        DetectorAttributes attrs;
        int locked_id = sqlite3_column_int(stmt, 0);
        attrs.size = static_cast<float>(sqlite3_column_double(stmt, 1));
        attrs.angle = static_cast<float>(sqlite3_column_double(stmt, 2));
        attrs.response = static_cast<float>(sqlite3_column_double(stmt, 3));
        attrs.octave = sqlite3_column_int(stmt, 4);
        attrs.class_id = sqlite3_column_int(stmt, 5);
        attributes_map.emplace(locked_id, attrs);
    }

    sqlite3_finalize(stmt);
    return attributes_map;
}

std::vector<std::tuple<int, std::string, std::string>> DatabaseManager::getAvailableKeypointSets() const {
    std::vector<std::tuple<int, std::string, std::string>> sets;
    if (!impl_->enabled || !impl_->db) return sets;

    const auto sql = R"(
        SELECT id, name, generation_method
        FROM keypoint_sets
        ORDER BY created_at DESC
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint sets query: " << sqlite3_errmsg(impl_->db) << std::endl;
        return sets;
    }

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);
        const auto name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const auto method = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        
        if (name && method) {
            sets.emplace_back(id, std::string(name), std::string(method));
        }
    }

    sqlite3_finalize(stmt);
    return sets;
}

std::optional<DatabaseManager::KeypointSetInfo> DatabaseManager::getKeypointSetInfo(const std::string& name) const {
    if (!impl_->enabled || !impl_->db) return std::nullopt;

    const auto sql = R"(
        SELECT id, name, generator_type, generation_method, dataset_path,
               max_features, boundary_filter_px, overlap_filtering, min_distance,
               source_set_a_id, source_set_b_id, tolerance_px, intersection_method
        FROM keypoint_sets
        WHERE name = ?
        LIMIT 1
    )";

    sqlite3_stmt* stmt = nullptr;
    if (int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr); rc != SQLITE_OK) {
        std::cerr << "Failed to query keypoint set info: " << sqlite3_errmsg(impl_->db) << std::endl;
        return std::nullopt;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);

    std::optional<KeypointSetInfo> info;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        KeypointSetInfo result;
        result.id = sqlite3_column_int(stmt, 0);
        result.name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        result.generator_type = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        result.generation_method = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        result.dataset_path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        result.max_features = sqlite3_column_int(stmt, 5);
        result.boundary_filter_px = sqlite3_column_int(stmt, 6);
        result.overlap_filtering = sqlite3_column_int(stmt, 7) != 0;
        result.min_distance = static_cast<float>(sqlite3_column_double(stmt, 8));
        result.source_set_a_id = sqlite3_column_type(stmt, 9) == SQLITE_NULL ? -1 : sqlite3_column_int(stmt, 9);
        result.source_set_b_id = sqlite3_column_type(stmt, 10) == SQLITE_NULL ? -1 : sqlite3_column_int(stmt, 10);
        result.tolerance_px = static_cast<float>(sqlite3_column_double(stmt, 11));

        if (const unsigned char* intersection_text = sqlite3_column_text(stmt, 12)) {
            result.intersection_method = reinterpret_cast<const char*>(intersection_text);
        }
        info = result;
    }

    sqlite3_finalize(stmt);
    return info;
}

std::vector<std::string> DatabaseManager::getScenesForSet(int keypoint_set_id) const {
    std::vector<std::string> scenes;
    if (!impl_->enabled || !impl_->db) return scenes;

    const auto sql = R"(
        SELECT DISTINCT scene_name
        FROM locked_keypoints
        WHERE keypoint_set_id = ?
        ORDER BY scene_name
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to query scenes for set: " << sqlite3_errmsg(impl_->db) << std::endl;
        return scenes;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        if (const auto text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))) scenes.emplace_back(text);
    }

    sqlite3_finalize(stmt);
    return scenes;
}

std::vector<std::string> DatabaseManager::getImagesForSet(int keypoint_set_id, const std::string& scene_name) const {
    std::vector<std::string> images;
    if (!impl_->enabled || !impl_->db) return images;

    const auto sql = R"(
        SELECT DISTINCT image_name
        FROM locked_keypoints
        WHERE keypoint_set_id = ? AND scene_name = ?
        ORDER BY image_name
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to query images for set: " << sqlite3_errmsg(impl_->db) << std::endl;
        return images;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        if (auto text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))) images.emplace_back(text);
    }

    sqlite3_finalize(stmt);
    return images;
}

std::vector<std::string> DatabaseManager::getDetectorsForSet(const int keypoint_set_id) const {
    std::vector<std::string> detectors;
    if (!impl_->enabled || !impl_->db) return detectors;

    const auto sql = R"(
        SELECT DISTINCT detector_type
        FROM keypoint_detector_attributes
        WHERE locked_keypoint_id IN (
            SELECT id FROM locked_keypoints WHERE keypoint_set_id = ?
        )
        ORDER BY detector_type
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to query detector types for set: " << sqlite3_errmsg(impl_->db) << std::endl;
        return detectors;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        if (auto text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))) {
            detectors.emplace_back(text);
        }
    }

    sqlite3_finalize(stmt);
    return detectors;
}

bool DatabaseManager::clearKeypointsForSet(int keypoint_set_id) const {
    if (!impl_->enabled || !impl_->db) return true;

    const auto sql = "DELETE FROM locked_keypoints WHERE keypoint_set_id = ?";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare clear keypoints statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to clear keypoints for set: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::updateKeypointSetStats(const KeypointSetStats& stats) const {
    if (!impl_->enabled || !impl_->db) return true;
    if (stats.keypoint_set_id < 0) {
        std::cerr << "Invalid keypoint_set_id provided for stats update" << std::endl;
        return false;
    }

    const auto sql = R"(
        UPDATE keypoint_sets
        SET detection_time_cpu_ms = ?,
            detection_time_gpu_ms = ?,
            total_generation_cpu_ms = ?,
            total_generation_gpu_ms = ?,
            intersection_time_ms = ?,
            avg_keypoints_per_image = ?,
            total_keypoints = ?,
            source_a_keypoints = ?,
            source_b_keypoints = ?,
            keypoint_reduction_pct = ?
        WHERE id = ?
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare keypoint set stats update: "
                  << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_double(stmt, 1, stats.detection_time_cpu_ms);
    sqlite3_bind_double(stmt, 2, stats.detection_time_gpu_ms);
    sqlite3_bind_double(stmt, 3, stats.total_generation_cpu_ms);
    sqlite3_bind_double(stmt, 4, stats.total_generation_gpu_ms);
    sqlite3_bind_double(stmt, 5, stats.intersection_time_ms);
    sqlite3_bind_double(stmt, 6, stats.avg_keypoints_per_image);
    sqlite3_bind_int(stmt, 7, stats.total_keypoints);
    sqlite3_bind_int(stmt, 8, stats.source_a_keypoints);
    sqlite3_bind_int(stmt, 9, stats.source_b_keypoints);
    sqlite3_bind_double(stmt, 10, stats.keypoint_reduction_pct);
    sqlite3_bind_int(stmt, 11, stats.keypoint_set_id);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to update keypoint set stats: "
                  << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::clearAllDetectorAttributesForSet(int keypoint_set_id) const {
    if (!impl_->enabled || !impl_->db) return true;

    const auto sql = R"(
        DELETE FROM keypoint_detector_attributes
        WHERE locked_keypoint_id IN (
            SELECT id FROM locked_keypoints WHERE keypoint_set_id = ?
        )
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare clear detector attributes statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, keypoint_set_id);

    rc = sqlite3_step(stmt);
    const bool success = (rc == SQLITE_DONE);
    if (!success) {
        std::cerr << "Failed to clear detector attributes for set: " << sqlite3_errmsg(impl_->db) << std::endl;
    }

    sqlite3_finalize(stmt);
    return success;
}

bool DatabaseManager::storeMatches(int experiment_id,
                                  const std::string& scene_name,
                                  const std::string& query_image,
                                  const std::string& train_image,
                                  const std::vector<cv::KeyPoint>& query_kps,
                                  const std::vector<cv::KeyPoint>& train_kps,
                                  const std::vector<cv::DMatch>& matches,
                                  const std::vector<bool>& correctness_flags) const {
    if (!impl_->enabled || !impl_->db) return true; // Success if disabled

    if (matches.size() != correctness_flags.size()) {
        std::cerr << "Match and correctness vectors must have same size" << std::endl;
        return false;
    }

    const auto sql = R"(
        INSERT INTO matches (experiment_id, scene_name, query_image, train_image,
                           query_keypoint_x, query_keypoint_y, train_keypoint_x, train_keypoint_y,
                           distance, match_confidence, is_correct_match)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare match insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    // Start transaction for performance
    sqlite3_exec(impl_->db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    bool success = true;
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& match = matches[i];

        // Validate keypoint indices
        if (match.queryIdx >= static_cast<int>(query_kps.size()) ||
            match.trainIdx >= static_cast<int>(train_kps.size())) {
            std::cerr << "Invalid keypoint indices in match " << i << std::endl;
            success = false;
            break;
        }

        const auto& query_kp = query_kps[match.queryIdx];
        const auto& train_kp = train_kps[match.trainIdx];

        sqlite3_bind_int(stmt, 1, experiment_id);
        sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 3, query_image.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 4, train_image.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 5, query_kp.pt.x);
        sqlite3_bind_double(stmt, 6, query_kp.pt.y);
        sqlite3_bind_double(stmt, 7, train_kp.pt.x);
        sqlite3_bind_double(stmt, 8, train_kp.pt.y);
        sqlite3_bind_double(stmt, 9, match.distance);
        sqlite3_bind_double(stmt, 10, match.distance); // Use distance as confidence for now
        sqlite3_bind_int(stmt, 11, correctness_flags[i] ? 1 : 0);

        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Failed to insert match " << i << ": " << sqlite3_errmsg(impl_->db) << std::endl;
            success = false;
            break;
        }

        sqlite3_reset(stmt);
    }

    sqlite3_finalize(stmt);

    if (success) {
        sqlite3_exec(impl_->db, "COMMIT", nullptr, nullptr, nullptr);
        std::cout << "Stored " << matches.size() << " matches for "
                  << scene_name << "/" << query_image << " -> " << train_image
                  << " (experiment " << experiment_id << ")" << std::endl;
    } else {
        sqlite3_exec(impl_->db, "ROLLBACK", nullptr, nullptr, nullptr);
    }

    return success;
}

std::vector<cv::DMatch> DatabaseManager::getMatches(int experiment_id,
                                                    const std::string& scene_name,
                                                    const std::string& query_image,
                                                    const std::string& train_image) const {
    if (!impl_->enabled || !impl_->db) return {};

    const char* sql = R"(
        SELECT query_keypoint_x, query_keypoint_y, train_keypoint_x, train_keypoint_y, distance
        FROM matches
        WHERE experiment_id = ? AND scene_name = ? AND query_image = ? AND train_image = ?
        ORDER BY distance ASC
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare match select statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return {};
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, query_image.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, train_image.c_str(), -1, SQLITE_STATIC);

    std::vector<cv::DMatch> matches;
    int match_idx = 0;

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        cv::DMatch match;
        match.queryIdx = match_idx; // Note: We don't store actual indices, so use sequential
        match.trainIdx = match_idx;
        match.distance = static_cast<float>(sqlite3_column_double(stmt, 4));
        match.imgIdx = 0;

        matches.push_back(match);
        match_idx++;
    }

    sqlite3_finalize(stmt);
    return matches;
}

bool DatabaseManager::storeVisualization(int experiment_id,
                                        const std::string& scene_name,
                                        const std::string& visualization_type,
                                        const std::string& image_pair,
                                        const cv::Mat& visualization_image,
                                        const std::string& metadata) const {
    if (!impl_->enabled || !impl_->db) return true; // Success if disabled

    if (visualization_image.empty()) {
        std::cerr << "Cannot store empty visualization image" << std::endl;
        return false;
    }

    // Encode image to PNG format for storage
    std::vector<uchar> encoded_image;
    std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 6}; // Medium compression

    if (!cv::imencode(".png", visualization_image, encoded_image, compression_params)) {
        std::cerr << "Failed to encode visualization image to PNG" << std::endl;
        return false;
    }

    const auto sql = R"(
        INSERT OR REPLACE INTO visualizations (experiment_id, scene_name, visualization_type,
                                             image_pair, image_data, image_format, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare visualization insert statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, visualization_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, image_pair.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_blob(stmt, 5, encoded_image.data(), static_cast<int>(encoded_image.size()), SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, "PNG", -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, metadata.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to insert visualization: " << sqlite3_errmsg(impl_->db) << std::endl;
        return false;
    }

    std::cout << "Stored " << visualization_type << " visualization for "
              << scene_name << "/" << image_pair << " (experiment " << experiment_id
              << ", size: " << encoded_image.size() << " bytes)" << std::endl;

    return true;
}

cv::Mat DatabaseManager::getVisualization(int experiment_id,
                                         const std::string& scene_name,
                                         const std::string& visualization_type,
                                         const std::string& image_pair) const {
    if (!impl_->enabled || !impl_->db) return cv::Mat();

    const char* sql = R"(
        SELECT image_data, image_format
        FROM visualizations
        WHERE experiment_id = ? AND scene_name = ? AND visualization_type = ? AND image_pair = ?
    )";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare visualization select statement: " << sqlite3_errmsg(impl_->db) << std::endl;
        return cv::Mat();
    }

    sqlite3_bind_int(stmt, 1, experiment_id);
    sqlite3_bind_text(stmt, 2, scene_name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, visualization_type.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, image_pair.c_str(), -1, SQLITE_STATIC);

    cv::Mat result;

    if ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const void* blob_data = sqlite3_column_blob(stmt, 0);
        int blob_size = sqlite3_column_bytes(stmt, 0);
        const char* format = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        if (blob_data && blob_size > 0) {
            // Decode image from stored format
            std::vector<uchar> encoded_data(static_cast<const uchar*>(blob_data),
                                          static_cast<const uchar*>(blob_data) + blob_size);

            result = cv::imdecode(encoded_data, cv::IMREAD_COLOR);

            if (result.empty()) {
                std::cerr << "Failed to decode visualization image from " << (format ? format : "unknown") << " format" << std::endl;
            }
        }
    }

    sqlite3_finalize(stmt);
    return result;
}

} // namespace thesis_project::database
