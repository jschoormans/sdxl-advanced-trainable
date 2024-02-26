#!/bin/bash

# DigitalOcean Spaces details
SPACE_NAME="tmp-lora-experiment-2"
SPACE_REGION="ams3"
SPACE_ENDPOINT="https://${SPACE_REGION}.digitaloceanspaces.com"

# Directory to search for 'training_out*' folders
SEARCH_DIR="./"

# Find and sync each 'training_out*' directory
find $SEARCH_DIR -type d -name "training_out*" -print0 | while IFS= read -r -d '' folder; do
    FOLDER_NAME=$(basename "$folder")
    echo "Syncing $FOLDER_NAME to DigitalOcean Space..."
    aws s3 sync "$folder" "s3://$SPACE_NAME/$FOLDER_NAME" --endpoint-url "$SPACE_ENDPOINT"
done

echo "Sync completed."
