
import torch
import torch.nn as nn
import yaml


class ClueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.blstm = nn.LSTM(
            input_size = 512,
            hidden_size = 768,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.linear = nn.Linear(768 * 2, 768)
        self.tanh = nn.Tanh()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x): # x->(B, T, 128)
        x, _ = self.blstm(x)           # (B, T, 1024) from bidirectional LSTM with hidden_size=512
        x = self.tanh(x)               # (B, T, 1024)
        x = self.linear(x)
        x = x.transpose(1, 2)          # (B, 1024, T)
        x = self.pool(x)               # (B, 1024, 1)
        x = x.squeeze(-1)              # (B, 1024)
        return x


if __name__ == "__main__":
    encoder = ClueEncoder()
    x = torch.randn(2, 156, 512)  
    y = encoder(x)
    print(f"Input: {x.shape} -> Output: {y.shape}") # (2, 1024)
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")



