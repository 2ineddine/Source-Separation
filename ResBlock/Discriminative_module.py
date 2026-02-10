from layerspp import ResnetBlockBigGANpp
from clue_embedding import ClueEncoder
from Discriminative_Unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange
from TSEEncoder import TSEEncoder



class DiscriminativeModule(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.clue_encoder = ClueEncoder()
        # TSEEncoder expects embedding dimension matching ClueEncoder output (1024)
        self.Tse_encoder = TSEEncoder(e=1024)
        
        self.Unet = UNet()

    def forward(self, *, y, clue):
        e = self.clue_encoder(x=clue)
        mix = self.Tse_encoder(y=y, e=e)
        print (mix.shape)
        xout1, xout2 = self.Unet(mix)
        return xout1, xout2


# test of the architecture 
if __name__ == "__main__":
    model = DiscriminativeModule()
    y = torch.randn(2, 2, 512, 256)  
    clue = torch.randn(2, 156, 128)  
    out1,out2 = model(y=y, clue=clue)
    out1,out2 = model(y=y, clue=clue)
    print(f"Input y: {y.shape}, clue: {clue.shape} -> Output1: {out1.shape}, Output2: {out2.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


