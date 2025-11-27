# Default hyperparameters for the irregular transformer model.
# Update here to keep training and inference in sync.
# eplanation for each parameter:
# d_model: dimension of the model (embedding size)
# nhead: number of attention heads
# nlayers: number of transformer encoder layers 
# num_freqs: number of Fourier features for positional encoding
MODEL_CFG = {"d_model": 120, "nhead": 3, "nlayers": 3, "num_freqs": 6}

# embed_dim must be divisible by num_heads
# layers is at maximum capacity
