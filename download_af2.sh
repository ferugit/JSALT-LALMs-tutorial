#!/bin/bash

# Removes the maximum file size
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

HF_TOKEN=$1
echo $HF_TOKEN
local_dir="models/audio_flamingo_2"

# Check if the local directory exists
if [ ! -d "$local_dir" ]; then
    echo "Directory $local_dir does not exist. Creating it."
    mkdir -p $local_dir
else
    echo "Directory $local_dir already exists. Skipping creation."
fi

# Downloasd the 3B model under the models/audio_flamingo_2 directory
python src/scripts/download_af2.py \
    --local_dir $local_dir \
    --hf_token $HF_TOKEN
