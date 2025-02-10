# main.py
import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Project")
    parser.add_argument(
        "--DEVICE",
        type=str,
        default="cuda" if config.DEVICE == "cuda" else "cpu",
        help="Device to run on, e.g., 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode: 'train' for training, 'inference' for text generation."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Override the default device in config.py
    config.DEVICE = args.DEVICE
    print(f"Using device: {config.DEVICE}")
    
    if args.mode == "train":
        import train
        train.main()
    else:
        import inference
        inference.main()

if __name__ == "__main__":
    main()
