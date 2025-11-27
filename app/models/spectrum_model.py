import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrumClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SpectrumClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        # the pooling layer makes the input smaller and reduces overfitting, and also reduces the computational load
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 50, 128)  # Adjust input size based on your data
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define the forward pass
        # Assuming input x has shape (batch_size, 1, 100)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Continuous positional embedding (Fourier features) ----
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, scale=1.0):
        super().__init__()
        # frequencies are geometrically spaced; tweak as needed
        self.register_buffer("freqs", scale * (2.0 ** torch.arange(num_freqs, dtype=torch.float32)))
        self.out_dim = 2 * num_freqs  # sin and cos

    def forward(self, x):
        # x: (B, N, 1) wavenumbers (any units; consider pre-normalizing)
        # returns: (B, N, 2*num_freqs)
        ang = 2 * math.pi * x * self.freqs.view(1, 1, -1)  # (B,N,num_freqs)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

# ---- Irregular-spectrum classifier ----
class IrregularSpectrumClassifier(nn.Module):
    """
    Inputs:
      x_wn: (B, N, 1)  -> wavenumbers (irregular per sample)
      y_i : (B, N, 1)  -> intensities
      key_padding_mask: (B, N) True for PAD positions (will be ignored)
    """
    def __init__(self, num_classes, d_model=128, nhead=4, nlayers=3, ff_mult=4, num_freqs=8):
        super().__init__()
        self.posenc = FourierPositionalEncoding(num_freqs=num_freqs, scale=1.0)

        d_pos = self.posenc.out_dim
        d_feat = 1  # intensity
        d_in = d_pos + d_feat

        self.input_proj = nn.Linear(d_in, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # lightweight attention pooling instead of simple mean
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))  # learned [CLS]-like query
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x_wn, y_i, key_padding_mask=None):
        # x_wn, y_i: (B,N,1)
        p = self.posenc(x_wn)             # (B,N,2F)
        h = torch.cat([p, y_i], dim=-1)   # (B,N,2F+1)
        h = self.input_proj(h)            # (B,N,D)

        # Transformer encoder with mask (True means "ignore")
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,N,D)

        # Attention pooling using a learned query
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)  # (B,1,D)
        # Provide key_padding_mask to attention as well
        pooled, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask)  # (B,1,D)
        pooled = pooled.squeeze(1)  # (B,D)

        return self.head(pooled)

# ---- Collate function to batch variable-length spectra ----
def collate_irregular(batch, pad_value=0.0):
    """
    batch: list of dicts like
      {"wn": 1D tensor [Ni], "intensity": 1D tensor [Ni], "label": int}
    Returns padded tensors and mask usable by the model.
    """
    wn_list = [b["wn"].view(-1, 1).float() for b in batch]
    it_list = [b["intensity"].view(-1, 1).float() for b in batch]
    y_list  = [b["label"] for b in batch]

    lengths = [t.size(0) for t in wn_list]
    Nmax = max(lengths)
    B = len(batch)

    x_wn = torch.full((B, Nmax, 1), fill_value=pad_value, dtype=torch.float32)
    y_i  = torch.full((B, Nmax, 1), fill_value=pad_value, dtype=torch.float32)
    mask = torch.ones(B, Nmax, dtype=torch.bool)  # True = PAD (ignored)

    for i, (w, it) in enumerate(zip(wn_list, it_list)):
        n = w.size(0)
        x_wn[i, :n] = w
        y_i[i, :n] = it
        mask[i, :n] = False  # valid positions

    labels = torch.tensor(y_list, dtype=torch.long)
    return x_wn, y_i, mask, labels

# ---- Example usage ----
if __name__ == "__main__":
    # Fake irregular batch: different wavenumber grids per sample
    ex_batch = [
        {"wn": torch.tensor([500., 510., 523., 540., 560.]),
         "intensity": torch.randn(5),
         "label": 2},
        {"wn": torch.tensor([120., 200., 350., 351., 900., 1500.]),
         "intensity": torch.randn(6),
         "label": 0},
        {"wn": torch.tensor([100., 101.5, 102.1]),
         "intensity": torch.randn(3),
         "label": 1},
    ]
    x_wn, y_i, mask, labels = collate_irregular(ex_batch)

    model = IrregularSpectrumClassifier(num_classes=3, d_model=128, nhead=4, nlayers=3, num_freqs=8)
    logits = model(x_wn, y_i, key_padding_mask=mask)
    print(logits.shape)  # (B, num_classes)
