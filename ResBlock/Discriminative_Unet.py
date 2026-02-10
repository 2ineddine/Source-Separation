
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from Transformer_block import BasicTransformerBlock


class Block1D(nn.Module):
    """Basic 1D convolutional block"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.Mish()
    
    def forward(self, x):
        x = self.conv(x )
        x = self.norm(x)
        x = self.act(x)
        return x 


class ResnetBlock1D(nn.Module):
    """ResNet block without time conditioning"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x )


class Downsample1D(nn.Module):
    """Downsampling by factor of 2"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling by factor of 2 using transpose convolution"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from Transformer_block import BasicTransformerBlock


class Block1D(nn.Module):
    """Basic 1D convolutional block"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.Mish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x 


class ResnetBlock1D(nn.Module):
    """ResNet block without time conditioning"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)
    
    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample1D(nn.Module):
    """Downsampling by factor of 2"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling by factor of 2"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Simplified 1D UNet: Single downsample → middle → single upsample
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        hidden_channels=128,
        dropout=0.05,
        attention_head_dim=8,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        act_fn="snakebeta",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.doubled_out_channels = out_channels * 2
        
        # ========== DOWNSAMPLE BLOCK ==========
        self.down_resnet = ResnetBlock1D(in_channels, hidden_channels)
        self.down_transformers = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_channels,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
            for _ in range(n_blocks)
        ])
        self.downsample = Downsample1D(hidden_channels)
        
        # ========== MIDDLE BLOCKS ==========
        self.mid_resnets = nn.ModuleList([
            ResnetBlock1D(hidden_channels, hidden_channels)
            for _ in range(num_mid_blocks)
        ])
        self.mid_transformers = nn.ModuleList([
            nn.ModuleList([
                BasicTransformerBlock(
                    dim=hidden_channels,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ])
            for _ in range(num_mid_blocks)
        ])
        
        # ========== UPSAMPLE BLOCK ==========
        self.up_resnet = ResnetBlock1D(2 * hidden_channels, in_channels)
        self.up_transformers = nn.ModuleList([
            BasicTransformerBlock(
                dim=in_channels,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
            for _ in range(n_blocks)
        ])
        self.upsample = Upsample1D(hidden_channels)
        
        # ========== FINAL PROJECTION ==========
        self.final_block = Block1D(in_channels, in_channels)
        self.final_proj = nn.Conv1d(in_channels, self.doubled_out_channels, 1)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, C, T) - input (e.g., B, 256, 512)
            mask: (B, 1, T) - mask
            
        Returns:
            output1: (B, out_channels, T)
            output2: (B, out_channels, T)
        """
        # ========== DOWNSAMPLE ==========
        # ResNet
        x = self.down_resnet(x)
        print ("[DOWN][RESNET1] : ", x.shape)
        
        # Transformers
        x = rearrange(x, "b c t -> b t c")
        for transformer in self.down_transformers:
            x = transformer(hidden_states=x)
        
        print ("[DOWN][RESNET1][TRANSFORMER] : ", x.shape)

        x = rearrange(x, "b t c -> b c t")
        
        # Save skip connection
        skip = x
        
        # Downsample
        x = self.downsample(x)
        print ("[DOWN][RESNET1][TRANSFORMER][DOWNSAMPLE] : ", x.shape)
        # ========== MIDDLE ==========
        for resnet, transformers in zip(self.mid_resnets, self.mid_transformers):
            # ResNet
            x = resnet(x)
            print ("[MIDDLE][RESNET] : ", x.shape)
            # Transformers
            x = rearrange(x, "b c t -> b t c")
            for transformer in transformers:
                x = transformer(hidden_states=x)
            print ("[MIDDLE][RESNET][TRANSFORMER] : ", x.shape)
            x = rearrange(x, "b t c -> b c t")
        
        # ========== UPSAMPLE ==========
        # Upsample first
        x = self.upsample(x)
        print ("[UP][UPSAMPLE] : ", x.shape)
        # Concatenate skip connection
        x = pack([x, skip], "b * t")[0]
        print ("[UP][CONCAT] : ", x.shape)
        
        # ResNet
        x = self.up_resnet(x)
        print("[UP][RESNET] : ", x.shape)
        # Transformers
        x = rearrange(x, "b c t -> b t c")
        for transformer in self.up_transformers:
            x = transformer(hidden_states=x)
        print("[UP][RESNET][TRANSFORMER] : ", x.shape)
        x = rearrange(x, "b t c -> b c t")
        
        # ========== FINAL PROJECTION ==========
        x = self.final_block(x)
        print("[FINAL][BLOCK] : ", x.shape)
        output = self.final_proj(x)
        print("[FINAL][PROJ] : ", output.shape)
        
        # Split into two outputs
        output1, output2 = torch.chunk(output, 2, dim=1)
        
        return output1, output2


# Example usage
if __name__ == "__main__":
    model = UNet(
        in_channels=512,
        out_channels=512,
        hidden_channels=128,
        dropout=0.1,
        attention_head_dim=8,
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=2,
        act_fn="snakebeta",
    )
    
    x = torch.randn(2, 512, 256)
    
    output1, output2 = model(x)
    print(f"Output1 shape: {output1.shape}")  # (4, 256, 512)
    print(f"Output2 shape: {output2.shape}")  # (4, 256, 512)
    
    pn = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pn:,}")