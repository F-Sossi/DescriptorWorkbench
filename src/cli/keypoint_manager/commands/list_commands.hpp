#pragma once

#include "thesis_project/database/DatabaseManager.hpp"

namespace thesis_project::cli::keypoint_commands {

int listSets(thesis_project::database::DatabaseManager& db);
int listScenes(thesis_project::database::DatabaseManager& db);
int countKeypoints(thesis_project::database::DatabaseManager& db, int argc, char** argv);
int listDetectors();

} // namespace thesis_project::cli::keypoint_commands
