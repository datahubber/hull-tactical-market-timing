#!/usr/bin/env bash
# Download the Hull Tactical competition files with the official Kaggle CLI.
# Needs a Kaggle account that has accepted the competition rules and API credentials set up
# as the Kaggle CLI documents. The data are not redistributed in this repository.
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
kaggle competitions download -c hull-tactical-market-prediction -p "$DEST"
unzip -o -q "$DEST/hull-tactical-market-prediction.zip" -d "$DEST"
rm -f "$DEST/hull-tactical-market-prediction.zip"
ls -l "$DEST/train.csv"
