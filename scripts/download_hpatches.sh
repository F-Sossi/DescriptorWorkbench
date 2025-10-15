#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
ARCHIVE_NAME="hpatches-sequences-release.tar.gz"
HPATCHES_URL="http://www.cvl.isy.liu.se/en/research/datasets/hpatches/hpatches-sequences-release.tar.gz"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f "$ARCHIVE_NAME" ]; then
    echo "Downloading HPatches dataset..."
    curl -L "$HPATCHES_URL" -o "$ARCHIVE_NAME"
else
    echo "Archive already present: $ARCHIVE_NAME"
fi

if [ ! -d "hpatches-sequences-release" ]; then
    echo "Extracting HPatches dataset..."
    tar -xf "$ARCHIVE_NAME"
else
    echo "Dataset directory already exists: hpatches-sequences-release"
fi

echo "HPatches ready at $(pwd)/hpatches-sequences-release"
