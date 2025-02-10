# data_utils.py
import os
import io
import requests
import zipfile
import torch
import sentencepiece as spm
from config import CONTEXT, BATCH_SIZE, DEVICE


def load_tokenizer(model_path='wiki_tokenizer.model'):
    """Load the SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp

def load_or_create_data(tokenizer, text_file='wiki.txt', data_file='encoded_data.pt'):
    """Load pre-encoded data or encode and save the data if not already saved."""
    if os.path.exists(data_file):
        print("Loading saved encoded data...")
        data = torch.load(data_file)
    else:
        print("Encoding data...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded = tokenizer.Encode(text)
        data = torch.tensor(encoded, dtype=torch.long)
        torch.save(data, data_file)
    return data

def get_train_val_data(data):
    """Split the data into training and validation sets."""
    data_size = len(data)
    split = int(0.9 * data_size)
    train_data = data[:split]
    val_data = data[split:]
    print(f"Total data: {data_size/1e6:.2f} Million | Training: {len(train_data)/1e6:.2f} Million | Validation: {len(val_data)/1e6:.2f} Million")
    return train_data, val_data

def get_batch(split, train_data, val_data):
    """Return a batch of data for training or validation."""
    data = train_data if split == "train" else val_data
    inds = torch.randint(len(data) - CONTEXT, (BATCH_SIZE,))
    x = torch.stack([data[i: i+CONTEXT] for i in inds])
    y = torch.stack([data[i+1: i+CONTEXT+1] for i in inds])
    return x.to(DEVICE), y.to(DEVICE)
