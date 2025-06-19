#!/bin/bash

# CUDA: check in the machine
export CUDA_VISIBLE_DEVICES=0

# Removes the maximum file size that can be created
ulimit -f unlimited

# Removes the maximum amount of virtual memory available
ulimit -v unlimited

# Limit the number of processes
ulimit -u 4096

# Not use remote data
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1


################################################
# Running configs
################################################

# Seed for reproducibility
SEED=2025

# Prompt for the model
prompt="Caption the audio."

# Input test and audio path
AUDIO_FILE="assets/dance_matisse_musiclm.wav"

# Config file
CONFIG_FILE="src/audio_flamingo_2/config/inference.yaml"


python -u src/scripts/single_inference_af2.py \
    --prompt "$prompt" \
    --audio_path $AUDIO_FILE \
    --seed $SEED \
    --config_path $CONFIG_FILE
