
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, pack, unpack

class ResnetBlockBigGANpp2D(nn.Module):
    """
    Minimal BigGAN++ ResNet block for 2D with embedding support
    Fits your TSEEncoder architecture
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        emb_dim=None,
        dropout=0.1,
        skip_rescale=True,
        use_depthwise=True,
    ):
        super().__init__()
        
        # Adaptive GroupNorm
        self.norm_0 = nn.GroupNorm(
            num_groups=max(1, min(in_ch // 4, 16)),
            num_channels=in_ch,
            eps=1e-6
        )
        
        self.act = nn.SiLU()
        
        # First conv - depthwise separable
        if use_depthwise and in_ch >= 8:
            self.conv_0 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=True)
            )
        else:
            self.conv_0 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # Embedding projection (optional)
        if emb_dim is not None:
            self.emb_proj = nn.Linear(emb_dim, out_ch)
        else:
            self.emb_proj = None
        
        self.norm_1 = nn.GroupNorm(
            num_groups=max(1, min(out_ch // 4, 16)),
            num_channels=out_ch,
            eps=1e-6
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Second conv
        if use_depthwise and out_ch >= 8:
            self.conv_1 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
                nn.Conv2d(out_ch, out_ch, 1, bias=True)
            )
        else:
            self.conv_1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # Skip connection
        if in_ch != out_ch:
            self.conv_skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.conv_skip = None
        
        self.skip_rescale = skip_rescale
    
    def forward(self, x, emb=None):
        h = self.act(self.norm_0(x))
        h = self.conv_0(h)
        
        # Add embedding if provided
        if emb is not None and self.emb_proj is not None:
            h = h + self.emb_proj(self.act(emb))[:, :, None, None]
        
        h = self.act(self.norm_1(h))
        h = self.dropout(h)
        h = self.conv_1(h)
        
        # Skip
        if self.conv_skip is not None:
            x = self.conv_skip(x)
        
        if self.skip_rescale:
            return (x + h) / np.sqrt(2.0)
        else:
            return x + h


class TSEEncoder(nn.Module):
    """
    Minimal TSEEncoder using BigGAN++ blocks
    Input: (B, 2, 512, 268) -> Output: (B, 256, 512)
    """
    def __init__(self, *, emb_dim=768, hidden_ch=16, use_depthwise=True):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Encoder blocks with embedding support
        self.block1 = ResnetBlockBigGANpp2D(2, hidden_ch, emb_dim=emb_dim, use_depthwise=use_depthwise)
        #self.block2 = ResnetBlockBigGANpp2D(hidden_ch, hidden_ch, emb_dim=emb_dim, use_depthwise=use_depthwise)
        self.block3 = ResnetBlockBigGANpp2D(hidden_ch, 1, emb_dim=None, use_depthwise=False)  # Final: no emb
        
        # Spatial adjustment: 268 -> 256
        self.spatial_adjust = nn.AdaptiveAvgPool2d((512, 256))
        
        # AdaIN conditioning
        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )
        self.gamma = nn.Linear(128, 512 * 256)
        self.beta = nn.Linear(128, 512 * 256)
    
    def forward(self, *, y, e):
        """
        Args:
            y: (B, 2, 512, 268)
            e: (B, emb_dim)
        Returns:
            (B, 256, 512)
        """
        b = y.shape[0]
        
        # Encoder with embedding conditioning
        y = self.block1(y, e)    # (B, 32, 512, 268)
        #y = self.block2(y, e)    # (B, 32, 512, 268)
        y = self.block3(y)       # (B, 1, 512, 268)
        
        # Spatial adjustment
        y = self.spatial_adjust(y)  # (B, 1, 512, 256)
        
        # AdaIN
        e_compressed = self.emb_mlp(e)
        gamma = self.gamma(e_compressed).view(b, 1, 512, 256)
        beta = self.beta(e_compressed).view(b, 1, 512, 256)
        y = y * gamma + beta
        
        return y.squeeze(1)  # (B, 512, 256)
    
#test TSEEncoder
if __name__ == "__main__":
    encoder = TSEEncoder()
    x = torch.randn(2, 2, 512,256)  
    e = torch.randn(2, 768)  
    y = encoder(y=x, e=e)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print (sum(p.numel() for p in encoder.parameters()if p.requires_grad))