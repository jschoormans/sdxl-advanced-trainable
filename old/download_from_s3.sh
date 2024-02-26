#!/bin/bash

# DigitalOcean Spaces details
SPACE_NAME="tmp-lora-experiment-2"
SPACE_REGION="ams3"
SPACE_ENDPOINT="https://${SPACE_REGION}.digitaloceanspaces.com"
# The bucket folder you want to sync, leave empty to sync the entire Space
BUCKET_FOLDER=""

# Local directory where files will be downloaded
LOCAL_DOWNLOAD_DIR="/home/ubuntu/sdxl-inpainting-trainable/trainings/"

# Ensure the local download directory exists
mkdir -p "$LOCAL_DOWNLOAD_DIR"

echo "Syncing from DigitalOcean Space to local directory..."
aws s3 sync "s3://$SPACE_NAME/$BUCKET_FOLDER" "$LOCAL_DOWNLOAD_DIR" --endpoint-url "$SPACE_ENDPOINT"

echo "Sync completed."
