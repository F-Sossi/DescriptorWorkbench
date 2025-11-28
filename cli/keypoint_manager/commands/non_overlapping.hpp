#pragma once

#include "thesis_project/database/DatabaseManager.hpp"

namespace thesis_project::cli::keypoint_commands {

int generateNonOverlapping(thesis_project::database::DatabaseManager& db, int argc, char** argv);

} // namespace thesis_project::cli::keypoint_commands
