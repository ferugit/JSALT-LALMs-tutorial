import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description="Download and cache Hugging Face model.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--cache_dir', type=str, default='models/', help='Directory to cache the model')
    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir

    # Download and cache the tokenizer and model locally
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

