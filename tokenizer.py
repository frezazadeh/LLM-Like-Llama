import sentencepiece as spm
import os

def train_tokenizer(input_file, model_prefix, vocab_size=4096):
    """
    Train a SentencePiece tokenizer on the provided text file.

    Args:
        input_file (str): Path to the input text file for training.
        model_prefix (str): Prefix for the output model files.
        vocab_size (int, optional): Desired size of the vocabulary. Defaults to 4096.
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=0.995,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
        train_extremely_large_corpus=False  # to avoid excessive memory usage
    )
    print(f"Tokenizer training completed. Model and vocab files saved as '{model_prefix}.model' and '{model_prefix}.vocab'.")

def load_tokenizer(model_file):
    """
    Load a trained SentencePiece tokenizer model.

    Args:
        model_file (str): Path to the trained SentencePiece model file.

    Returns:
        spm.SentencePieceProcessor: Loaded tokenizer instance.
    """
    tokenizer = spm.SentencePieceProcessor(model_file=model_file)
    print(f"Tokenizer loaded successfully from '{model_file}'.")
    return tokenizer

def test_tokenizer(tokenizer, text):
    """
    Test the tokenizer by encoding and decoding a sample text.

    Args:
        tokenizer (spm.SentencePieceProcessor): Loaded tokenizer instance.
        text (str): Text to encode and decode.
    """
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print("Original Text:", text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

# Example Usage
if __name__ == "__main__":
    input_file_path = "wiki.txt"
    model_prefix = "wiki_tokenizer"
    vocabulary_size = 1294

    # Train the tokenizer
    train_tokenizer(input_file_path, model_prefix, vocabulary_size)

    # Load the trained tokenizer
    tokenizer = load_tokenizer(f"{model_prefix}.model")

    # Print vocabulary size
    vocab_size = tokenizer.get_piece_size()
    print(f"Vocabulary Size: {vocab_size}")

    # Test the tokenizer
    sample_text = "What is 5G technology?"
    test_tokenizer(tokenizer, sample_text)
