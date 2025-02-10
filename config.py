# config.py
import torch

# Architecture parameters
BATCH_SIZE = 8
CONTEXT = 512
EMBED_SIZE = 384
N_LAYERS = 7
N_HEADS = 7
BIAS = True

# Hyperparameters
LEARNING_RATE = 3e-4
DROPOUT = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Training parameters
TRAIN_ITERS = 100000
EVAL_INTERVAL = 50
EVAL_ITERS = 3
COMPILE = False
LOAD_PRETRAINED = False

# Checkpoint settings
CHECKPOINT_DIR = 'models/'
CHECKPOINT_FN = "latest.pt"
CHECKPOINT_LOAD_FN = "latest.pt"

# Default device and precision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Logging settings (if you wish to use wandb)
WANDB_LOG = False  # Set to True if you use wandb
WANDB_PROJECT = "test"

# Inference mode flag (not used directly here)
INFERENCE = False
