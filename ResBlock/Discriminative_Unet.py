
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



# import math
# import torch
# from torch import nn
# from torch.nn import init
# from torch.nn import functional as F


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         time = time.float()
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings


# class TimeEmbedding(nn.Module):
#     def __init__(self, T, d_model, dim):
#         assert d_model % 2 == 0
#         super().__init__()
#         emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
#         emb = torch.exp(-emb)
#         pos = torch.arange(T).float()
#         emb = pos[:, None] * emb[None, :]
#         assert list(emb.shape) == [T, d_model // 2]
#         emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
#         assert list(emb.shape) == [T, d_model // 2, 2]
#         emb = emb.view(T, d_model)

#         self.timembedding = nn.Sequential(
#             nn.Embedding.from_pretrained(emb),
#             nn.Linear(d_model, dim),
#             Swish(),
#             nn.Linear(dim, dim),
#         )
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)

#     def forward(self, t):
#         emb = self.timembedding(t)
#         return emb


# class DownSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)

#     def forward(self, x, temb):
#         x = self.main(x)
#         return x


# class UpSample(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.main.weight)
#         init.zeros_(self.main.bias)

#     def forward(self, x, temb):
#         _, _, H, W = x.shape
#         x = F.interpolate(x, scale_factor=2, mode="nearest")
#         x = self.main(x)
#         return x


# class AttnBlock(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.group_norm = nn.GroupNorm(32, in_ch)
#         self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
#         self.initialize()

#     def initialize(self):
#         for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
#             init.xavier_uniform_(module.weight)
#             init.zeros_(module.bias)
#         init.xavier_uniform_(self.proj.weight, gain=1e-5)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         h = self.group_norm(x)
#         q = self.proj_q(h)
#         k = self.proj_k(h)
#         v = self.proj_v(h)

#         q = q.permute(0, 2, 3, 1).view(B, H * W, C) # 
#         k = k.view(B, C, H * W)
#         w = torch.bmm(q, k) * (int(C) ** (-0.5))
#         assert list(w.shape) == [B, H * W, H * W]
#         w = F.softmax(w, dim=-1)

#         v = v.permute(0, 2, 3, 1).view(B, H * W, C)
#         h = torch.bmm(w, v)
#         assert list(h.shape) == [B, H * W, C]
#         h = h.view(B, H, W, C).permute(0, 3, 1, 2)
#         h = self.proj(h)

#         return x + h


# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.GroupNorm(32, in_ch),
#             Swish(),
#             nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
#         )
#         self.temb_proj = nn.Sequential(
#             Swish(),
#             nn.Linear(tdim, out_ch),
#         )
#         self.block2 = nn.Sequential(
#             nn.GroupNorm(32, out_ch),
#             Swish(),
#             nn.Dropout(dropout),
#             nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
#         )
#         if in_ch != out_ch:
#             self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
#         else:
#             self.shortcut = nn.Identity()
#         if attn:
#             self.attn = AttnBlock(out_ch)
#         else:
#             self.attn = nn.Identity()
#         self.initialize()

#     def initialize(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 init.xavier_uniform_(module.weight)
#                 init.zeros_(module.bias)
#         init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

#     def forward(self, x, temb):
#         h = self.block1(x)
#         h += self.temb_proj(temb)[:, :, None, None]
#         h = self.block2(h)

#         h = h + self.shortcut(x)
#         h = self.attn(h)
#         return h


# class UNet(nn.Module):
#     def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
#         super().__init__()
#         assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
#         tdim = ch * 4
#         # self.time_embedding = TimeEmbedding(T, ch, tdim)
#         self.time_embedding = nn.Sequential(
#             SinusoidalPositionEmbeddings(tdim),
#             nn.Linear(tdim, tdim * 4),
#             nn.SiLU(),
#             nn.Linear(tdim * 4, tdim),
#         )

#         self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
#         self.downblocks = nn.ModuleList()
#         chs = [ch]  # record output channel when dowmsample for upsample
#         now_ch = ch
#         for i, mult in enumerate(ch_mult):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks):
#                 self.downblocks.append(
#                     ResBlock(
#                         in_ch=now_ch,
#                         out_ch=out_ch,
#                         tdim=tdim,
#                         dropout=dropout,
#                         attn=(i in attn),
#                     )
#                 )
#                 now_ch = out_ch
#                 chs.append(now_ch)
#             if i != len(ch_mult) - 1:
#                 self.downblocks.append(DownSample(now_ch))
#                 chs.append(now_ch)

#         self.middleblocks = nn.ModuleList(
#             [
#                 ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
#                 ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
#             ]
#         )

#         self.upblocks = nn.ModuleList()
#         for i, mult in reversed(list(enumerate(ch_mult))):
#             out_ch = ch * mult
#             for _ in range(num_res_blocks + 1):
#                 self.upblocks.append(
#                     ResBlock(
#                         in_ch=chs.pop() + now_ch,
#                         out_ch=out_ch,
#                         tdim=tdim,
#                         dropout=dropout,
#                         attn=(i in attn),
#                     )
#                 )
#                 now_ch = out_ch
#             if i != 0:
#                 self.upblocks.append(UpSample(now_ch))
#         assert len(chs) == 0

#         self.tail = nn.Sequential(
#             nn.GroupNorm(32, now_ch),
#             Swish(),
#             nn.Conv2d(now_ch, 3, 3, stride=1, padding=1),
#         )
#         self.initialize()

#     def initialize(self):
#         init.xavier_uniform_(self.head.weight)
#         init.zeros_(self.head.bias)
#         init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
#         init.zeros_(self.tail[-1].bias)

#     def forward(self, x, t):
#         # Timestep embedding
#         temb = self.time_embedding(t)
#         # Downsampling
#         h = self.head(x)
#         hs = [h]
#         for layer in self.downblocks:
#             h = layer(h, temb)
#             hs.append(h)
#         # Middle
#         for layer in self.middleblocks:
#             h = layer(h, temb)
#         # Upsampling
#         for layer in self.upblocks:
#             if isinstance(layer, ResBlock):
#                 h = torch.cat([h, hs.pop()], dim=1)
#             h = layer(h, temb)
#         h = self.tail(h)

#         assert len(hs) == 0
#         return h


# if __name__ == "__main__":
#     batch_size = 8
#     model = UNet(
#         T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1
#     )
#     x = torch.randn(batch_size, 3, 32, 32)
#     t = torch.randint(1000, (batch_size,))
#     y = model(x, t)
#     print(y.shape)
#     np = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {np:,}")