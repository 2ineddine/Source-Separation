
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, pack, unpack
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np

# ------------------------
# ResNet Block for (B, F, T)
# ------------------------
class ResnetBlock1D(nn.Module):
    def __init__(self, in_f, out_f, emb_dim=None, dropout=0.1):
        super().__init__()
        self.norm_0 = nn.LayerNorm(in_f)          # Normalize along F
        self.act = nn.SiLU()
        self.conv_0 = nn.Conv1d(in_f, out_f, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_f) if emb_dim is not None else None
        self.norm_1 = nn.LayerNorm(out_f)
        self.dropout = nn.Dropout(dropout)
        self.conv_1 = nn.Conv1d(out_f, out_f, 3, padding=1)
        self.skip = nn.Conv1d(in_f, out_f, 1) if in_f != out_f else None

    def forward(self, x, emb=None):
        # x: (B, F, T)
        h = self.act(self.norm_0(x.transpose(1,2))).transpose(1,2)  # LayerNorm along F
        h = self.conv_0(h)
        if emb is not None and self.emb_proj is not None:
            h = h + self.emb_proj(self.act(emb))[:, :, None]  # Broadcast along T
        h = self.act(self.norm_1(h.transpose(1,2))).transpose(1,2)
        h = self.dropout(h)
        h = self.conv_1(h)
        if self.skip is not None:
            x = self.skip(x)
        return (x + h) / np.sqrt(2.0)

# ------------------------
# TSE Encoder
# ------------------------
class TSEEncoder1D(nn.Module):
    def __init__(self, *, emb_dim=768, out_f=512):
        super().__init__()

        self.block1 = ResnetBlock1D(in_f=out_f, out_f=out_f)

        # Embedding conditioning (AdaIN-like)
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )
        self.gamma = nn.Linear(64, out_f)
        self.beta = nn.Linear(64, out_f)

    def forward(self, *, y, e):
        """
        y: (B, F, T)
        e: (B, emb_dim)
        returns: (B, out_f, T)
        """
        y = self.block1(y, e)
    

        # Conditioning
        e_c = self.emb_mlp(e)
        gamma = self.gamma(e_c).unsqueeze(-1)  # (B, out_f, 1)
        beta = self.beta(e_c).unsqueeze(-1)
        y = y * gamma + beta
        return y





#test TSEEncoder
if __name__ == "__main__":
    encoder = TSEEncoder1D()
    x = torch.randn(2, 512,256)  
    e = torch.randn(2, 768)  
    y = encoder(y=x, e=e)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print (sum(p.numel() for p in encoder.parameters()if p.requires_grad))