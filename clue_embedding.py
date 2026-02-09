
import torch
import torch.nn as nn
import yaml


class ClueEncoder(nn.Module):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = config['clue_encoder']
        
        self.blstm = nn.LSTM(
            input_size=params['input_dim'],
            hidden_size=params['hidden_dim'],
            num_layers=params['num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=params['dropout']
        )
        
        self.linear = nn.Linear(params['hidden_dim'] * 2, params['hidden_dim'])
        self.tanh = nn.Tanh()
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x, _ = self.blstm(x)           # (B, T, 2048)
        #print("After BLSTM:", x.shape)
        x = self.linear(x)             # (B, T, 1024)
        #print("After Linear:", x.shape)
        x = self.tanh(x)               # (B, T, 1024)
        #print("After Tanh:", x.shape)
        x = x.transpose(1, 2)          # (B, 1024, T)
        #print("After Transpose:", x.shape)
        x = self.pool(x)               # (B, 1024, 1)
        #print("After Pooling:", x.shape)
        x = x.squeeze(-1)              # (B, 1024)
        return x


if __name__ == "__main__":
    encoder = ClueEncoder()
    x = torch.randn(1, 156, 80)  
    y = encoder(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    # the number of parameters 
    
    print (f"Number of parameters: {sum(p.numel() for p in encoder.parameters())}")



