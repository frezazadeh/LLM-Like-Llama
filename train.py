# train.py
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import config
from config import DEVICE, DTYPE, TRAIN_ITERS, EVAL_INTERVAL, EVAL_ITERS, CHECKPOINT_DIR, CHECKPOINT_FN, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, LOAD_PRETRAINED, COMPILE, WANDB_LOG, WANDB_PROJECT
from data_utils import load_tokenizer, load_or_create_data, get_train_val_data, get_batch
from model import GPT

if WANDB_LOG:
    import wandb

def main():
    # Initialize wandb if logging is enabled
    if WANDB_LOG:
        wandb_run_name = "train-run-" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        wandb.init(project=WANDB_PROJECT, name=wandb_run_name)

    # Prepare data and tokenizer
    tokenizer = load_tokenizer()
    data = load_or_create_data(tokenizer)
    train_data, val_data = get_train_val_data(data)
    vocab_size = tokenizer.get_piece_size()
    
    # Instantiate the model
    model = GPT(vocab_size)
    model = model.to(DTYPE).to(DEVICE)
    
    if COMPILE:
        print("Compiling model...")
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model has {total_params:.2f} million parameters")
    
    # Setup optimizer and scheduler
    p_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]
    no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]
    optimizer_groups = [
        {'params': weight_decay_p, 'weight_decay': WEIGHT_DECAY},
        {'params': no_weight_decay_p, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=LEARNING_RATE, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAIN_ITERS, eta_min=LEARNING_RATE/10)
    
    # Optionally load a checkpoint
    start_iteration = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FN)
    if os.path.exists(checkpoint_path) and LOAD_PRETRAINED:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration']
        best_val_loss = checkpoint['loss']
        print(f"Loaded checkpoint from iteration {start_iteration} with loss {best_val_loss}")
    
    def calculate_loss():
        """Evaluate the model on a few batches for both training and validation splits."""
        losses = {}
        model.eval()
        for split in ['train', 'eval']:
            loss_vals = torch.zeros(EVAL_ITERS)
            for i in range(EVAL_ITERS):
                xb, yb = get_batch(split, train_data, val_data)
                _, loss = model(xb, yb)
                loss_vals[i] = loss.item()
            losses[split] = loss_vals.mean().item()
        model.train()
        return losses
    
    # Main training loop
    try:
        for i in tqdm(range(start_iteration, TRAIN_ITERS)):
            xb, yb = get_batch("train", train_data, val_data)
            _, loss = model(xb, yb)
            
            if i % EVAL_INTERVAL == 0 or i == TRAIN_ITERS - 1:
                losses = calculate_loss()
                print(f"\nIteration {i}: train loss: {losses['train']:.4f}, val loss: {losses['eval']:.4f}")
                
                # Generate a sample to check progress
                prompt = "The mountain in my city is"
                input_ids = torch.tensor(tokenizer.Encode(prompt), dtype=torch.long, device=DEVICE)[None, :]
                sample_ids = model.generate(input_ids, max_new_tokens=64)[0].tolist()
                sample_text = tokenizer.Decode(sample_ids)
                print(f"Sample generated: {sample_text}")
                
                if losses['eval'] < best_val_loss:
                    best_val_loss = losses['eval']
                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                        'iteration': i,
                    }, checkpoint_path)
                    print(f"[Checkpoint] Saved at iteration {i} with val loss {best_val_loss}")
                    
                if WANDB_LOG:
                    wandb.log({
                        "loss/train": losses['train'],
                        "loss/val": losses['eval'],
                        "lr": scheduler.get_last_lr()[0]
                    }, step=i)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            scheduler.step()
    
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        torch.cuda.empty_cache()
        print("GPU memory released.")
    
    if WANDB_LOG:
        wandb.finish()

if __name__ == "__main__":
    main()
