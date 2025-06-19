import argparse
from huggingface_hub import snapshot_download


def main(args):

    # Download the model
    snapshot_download(
        repo_id="nvidia/audio-flamingo-2-0.5B",
        local_dir=args.local_dir,
        token=args.hf_token
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download the Audio Flamingo 2 model.")
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token for authentication",
        required=True,
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        help="Local directory to save the model",
        required=True,
    )
    args = parser.parse_args()

    main(args)
