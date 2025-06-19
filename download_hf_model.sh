#!/bin/bash

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

MODEL_NAME="Qwen/Qwen2.5-0.5B"
CACHE_DIR="../models/"

# Download the model
python -u ../src/scripts/download_hf_model.py \
    --model_name $MODEL_NAME \
    --cache_dir $CACHE_DIR 