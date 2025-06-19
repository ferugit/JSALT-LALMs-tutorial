import sys
sys.path.insert(1, sys.path[0].replace(sys.path[0].split('/')[-1], ''))

import os
import yaml
import json
import random
import argparse

import torch
import librosa
import numpy as np
from pydub import AudioSegment

import audio_flamingo_2.factory as factory
from audio_flamingo_2.inference_utils import read_audio, load_audio, predict, get_num_windows
from audio_flamingo_2.utils import Dict2Class, float32_to_int16, int16_to_float32, get_autocast, get_cast_dtype
from safetensors.torch import load_file


def main(args):

    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)

    # Do not allow HF to connect to the internet
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid warnings

    # Load the config file
    config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

    #print(config)
    data_config = config['data_config']
    model_config = config['model_config']
    clap_config = config['clap_config']
    model_args = Dict2Class(config['train_config'])

    # Cast the model to the appropriate dtype
    autocast = get_autocast(
        model_args.precision, cache_enabled=(not model_args.fsdp)
    )
    cast_dtype = get_cast_dtype(model_args.precision)

    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set Hugging Face cache directory
    model, tokenizer = factory.create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=True,
        gradient_checkpointing=False,
        freeze_lm_embeddings=True,
        device=device,
    )

    print("Model and tokenizer created successfully.")
    print(model)

    print("Loading trained weights...")

    # CLAP, tokenizer and LLM are pretrained. 
    # XATTN and Transformer are not. We need to load the pretrained weights.
    model = model.to(device)
    model.eval()

    # Load the pretrained weights
    ckpt_path = config['inference_config']['pretrained_path']
    metadata_path = os.path.join(ckpt_path, "safe_ckpt/metadata.json")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct the full state_dict
    state_dict = {}

    # Load each SafeTensors chunk
    for chunk_name in metadata:
        chunk_path = f"safe_ckpt/{chunk_name}.safetensors"
        chunk_tensors = load_file(os.path.join(ckpt_path, chunk_path))

        # Merge tensors into state_dict
        state_dict.update(chunk_tensors)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    print("Model loaded successfully.")

    # Perform inference
    result = predict(
        args.audio_path,
        args.prompt,
        clap_config,
        inference_kwargs={
            "do_sample": False,  # Set to True for sampling, False for greedy/beam search
            "temperature": 0.0,
            "num_beams": 1,
            "top_k": 30,
            "top_p": 0.95,
            "num_return_sequences": 1,
        },
        cast_dtype=cast_dtype,
        device=device,
        tokenizer=tokenizer,
        model=model
    )
    print("Inference completed.\n\n")
    print("*" * 50)
    print("Prompt:", args.prompt)
    print("Audio path:", args.audio_path)
    print("Inference result:", result)


if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser("Single inference using AF2")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to use for inference",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file to use for inference",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )

    # Parse arguments
    args = parser.parse_args()

    main(args)