#pragma once

#include "thesis_project/database/DatabaseManager.hpp"

namespace thesis_project::cli::keypoint_commands {

int generateRandomSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv);
int generateTopNSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv);
int generateSpatialSubset(thesis_project::database::DatabaseManager& db, int argc, char** argv);

} // namespace thesis_project::cli::keypoint_commands
