# inference.py
import torch
from config import DEVICE, DTYPE, CHECKPOINT_DIR, CHECKPOINT_FN
from data_utils import load_tokenizer
from model import GPT

def main():
    print(f"Running inference on device: {DEVICE}")
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_piece_size()
    checkpoint_path = CHECKPOINT_DIR + CHECKPOINT_FN
    model = GPT(vocab_size)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DTYPE).to(DEVICE)
    model.eval()
    
    while True:
        prompt = input("Enter prompt (or 'q' to quit): ")
        if prompt.strip().lower() == "q":
            break
        input_ids = torch.tensor(tokenizer.Encode(prompt), dtype=torch.long, device=DEVICE)[None, :]
        generated_ids = model.generate(input_ids, max_new_tokens=64)[0].tolist()
        generated_text = tokenizer.Decode(generated_ids)
        print("Generated text:")
        print(generated_text)
        print()

if __name__ == "__main__":
    main()
