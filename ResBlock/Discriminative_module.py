from layerspp import ResnetBlockBigGANpp
from clue_embedding import ClueEncoder
from Unet import Unet
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiscriminativeModule(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, temb_dim=256):
        super(DiscriminativeModule, self).__init__()
        self.resblock = ResnetBlockBigGANpp(
            act=nn.SiLU(),
            in_ch=in_channels,
            out_ch=out_channels,
            temb_dim=temb_dim,
            up=False,
            down=True,
            skip_rescale=True
        )
        self.clue_encoder = ClueEncoder()
        self.unet = Unet(
            in_channels=out_channels,
            out_channels=out_channels,
            temb_dim=temb_dim
        )

    def forward(self, x):
        clue_embedding = self.clue_encoder(x)
        x = self.resblock(x, clue_embedding)
        x = self.unet(x, clue_embedding)
        return x





