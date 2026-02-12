import torch 
import torch.nn as nn

from UnetSDE import Unet
from  Discriminative_module import DiscriminativeModule





class DDPM (nn.Module):
    def __init__(self,*,stifness = 2,sigma0 = 0.05, sigma1 = 0.5):
        super().__init__()
        self.stifness = stifness
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.unet = Unet(in_channels=1024, out_channels=512) 
        self.discriminative_module = DiscriminativeModule()

        
    def add_noise (self,*, x,y, t):
        """Add noise to the input clean target x at time t, y is the cleaned mixture from the input."""
        
        mu = torch.exp(-self.stifness * t) * x + (1 - torch.exp(-self.stifness * t)) * y
        sigma = self.compute_sigma(t)

        z = torch.randn_like(x)
        xt = mu + sigma * z
        
        
        return xt,z
        
    def compute_sigma(self, t):
        
        log_ratio = torch.log(torch.tensor(self.sigma1 / self.sigma0))
        numerator = (self.sigma0**2 * 
                    ((self.sigma1/self.sigma0)**(2*t) - torch.exp(-2*self.stifness*t)) *  
                    log_ratio)
        denominator = self.stifness + log_ratio
        return torch.sqrt(numerator / denominator)


    def forward(self,*,x,y,t,clue):
        
        x_hat = self.discriminative_module(y=y, clue=clue)
        SNR_loss = -10*torch.log10(torch.mean(x**2) / torch.mean((x-x_hat)**2))
    
        x_hat = torch.mean(x_hat, dim=1, keepdim=True).squeeze(1)
        x = torch.mean(x, dim=1, keepdim=True).squeeze(1)  
        y = torch.mean(y, dim=1, keepdim=True).squeeze(1)
        print ("x_hat shape:", x_hat.shape, "x shape:", x.shape, "y shape:", y.shape)

        x_noised,z = self.add_noise(x=x,y=y,t=t)        
        sigma = self.compute_sigma(t)
        t = t.squeeze(-1).squeeze(-1)
        #t = t[:, None]
        print("the shape of t:", t.shape)
        score_pred = self.unet(x=x_noised, mu=x_hat, t=t)
        print("the shape of score_pred:", score_pred.shape)
        
        target = -z / sigma
        t_mask = t > 0.999
        if t_mask.any():
            target[t_mask] += -(x[t_mask] - y[t_mask]) * torch.exp(-self.stifness * \
                t[t_mask].view(-1,1,1)) / sigma[t_mask]**2 
        
        
        diffusion_loss = torch.mean((score_pred - target)**2)
        return diffusion_loss + SNR_loss
            
    
 
    

    
    @torch.no_grad()
    def synthese(self,*,clue,y, n_steps):
        
              
        x_hat = self.discriminative_module(y=y, clue=clue) 
 
        


        dt  = 1/n_steps
        J = []
        
        for j in range(10): # inference repetition
            
            sigma_T = self.compute_sigma(torch.tensor(1.0))  
            xt = y + sigma_T * torch.randn_like(y)  

            for step in range(n_steps):
                t = 1 - step * dt
                
                g = (self.sigma0 * ((self.sigma1 / self.sigma0))**t) * \
                    torch.sqrt(2*torch.log(self.sigma1 / self.sigma0))
        
                diffusion = g * torch.randn_like(xt)
                
                score = self.unet(x=xt, mask=None, mu=x_hat, t=t)
                
                f  = -self.stifness * ( xt - y)

                drift = -f + g**2 * score
                
                xt +=drift *dt + diffusion * torch.sqrt(dt)
                
                
                
               
                
            J.append(xt)
            
        return torch.stack(J, dim=0).mean(dim=0)
    
    
    



#test of the architecture
if __name__ == "__main__":
    model = DDPM()
    y = torch.randn(2, 2, 512, 256)  
    clue = torch.randn(2, 128, 512)  
    t = torch.randn(2).unsqueeze(-1).unsqueeze(-1)
    
    out = model(y=y, clue=clue, x=torch.randn(2,2, 512, 256), t=t)
    print(f"Input y: {y.shape}, clue: {clue.shape} -> Output: {out.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")