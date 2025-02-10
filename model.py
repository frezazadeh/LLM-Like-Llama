# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBED_SIZE, CONTEXT, N_LAYERS, N_HEADS, BIAS, DROPOUT, DEVICE

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBED_SIZE)
        self.positions = nn.Embedding(CONTEXT, EMBED_SIZE)
        self.blocks = nn.Sequential(*[Block(N_HEADS) for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(EMBED_SIZE)
        self.final_linear = nn.Linear(EMBED_SIZE, vocab_size, bias=BIAS)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input, targets=None):
        BS, SL = input.shape
        emb = self.embeddings(input)  # (BS, SL, EMBED_SIZE)
        pos = self.positions(torch.arange(SL, device=DEVICE))
        x = emb + pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.final_linear(x)
        loss = None
        if targets is not None:
            BS, SL, VS = logits.shape
            logits = logits.view(BS * SL, VS)
            targets = targets.view(BS * SL)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, input, max_new_tokens=500):
        for _ in range(max_new_tokens):
            input = input[:, -CONTEXT:]
            logits, _ = self(input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input = torch.cat((input, next_token), dim=1)
        return input

class Block(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        head_size = EMBED_SIZE // n_heads
        self.ma = Multihead(n_heads, head_size)
        self.feed_forward = ForwardLayer(EMBED_SIZE)
        self.ln1 = nn.LayerNorm(EMBED_SIZE)
        self.ln2 = nn.LayerNorm(EMBED_SIZE)
        
    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class ForwardLayer(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_size, 6 * embed_size, bias=BIAS),
            nn.GELU(),
            nn.Linear(6 * embed_size, embed_size, bias=BIAS),
            nn.Dropout(DROPOUT)
        )
        
    def forward(self, x):
        return self.network(x)

class Multihead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.combine = nn.Linear(head_size * n_heads, EMBED_SIZE, bias=BIAS)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.combine(x)
        x = self.dropout(x)
        return x

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.queries = nn.Linear(EMBED_SIZE, head_size, bias=BIAS)
        self.keys = nn.Linear(EMBED_SIZE, head_size, bias=BIAS)
        self.values = nn.Linear(EMBED_SIZE, head_size, bias=BIAS)
        # Create a lower triangular matrix for masking future tokens
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT, CONTEXT)))
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x):
        BS, SL, _ = x.shape
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        
        attn_weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        attn_weights = attn_weights.masked_fill(self.tril[:SL, :SL] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v
        return out
