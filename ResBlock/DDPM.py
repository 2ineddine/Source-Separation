import torch 
import torch.nn as nn

from ResBlock.Discriminative_Unet import UNet




class DDPM (nn.Module):
    def __init__(self,*,stifness = 2,sigma0 = 0.05, sigma1 = 0.5,n_steps =20 ):
        super().__init__()
        self.stifness = stifness
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.forward_unet = UNet(in_channels=512, out_channels=512)
        self.reverse_unet = UNet(in_channels=512, out_channels=512) 
        self.N = n_steps # number of steps for the reverse process, 
        self.sigma = torch.sqrt(((self.sigma0 ** 2) * ((self.sigma1  /self.sigma0 )**(2*t) * (torch.exp(-2 * self.stifness * t)) * torch.log(self.sigma1 / self.sigma0) ))/ (self.stifness+torch.log(self.sigma1/self.sigma0)))

        
        
        
        
    
    def forward(self,*,x_hat,x,t):
        
        mu = torch.exp(-self.stifness * t) * x + (1 - torch.exp(-self.stifness * t)) * x_hat
        
        z = torch.randn_like(x)
        xt = mu + self.sigma * z
        loss = torch.mean ((self.forward_unet(x=xt, mask=None, mu=mu, t=t) + z/self.sigma + (x-x_hat)*torch.exp(-self.stifness * t)/self.sigma**2)**2)
        return 
    
    def reverse (self,*,Unet,x_hat,xt,t, dt):
        f  = -self.stifness * ( x_hat - xt)
        
        g = (self.sigma0 * ((self.sigma1 / self.sigma0))**t) * torch.sqrt(2*torch.log(self.sigma1 / self.sigma0))
        diffusion = g * torch.randn_like(x_hat)
        score = Unet(x_hat,xt,t) 

        drift = -f * g**2 * score
        
        xt +=drift *dt + diffusion * torch.sqrt(dt)
        return xt
    
    @torch.no_grad()
    def synthese(self,*,Unet,x_hat,t, dt, n_steps):# repetitions=10) don't forget the inference repetion
        # and then you mean it to get a better result
        dt  = 1/self.N
        xt  = x_hat + torch.sqrt (self.drift)**2 * torch.randn_like(x_hat) 
      
        
        for step in range(n_steps):
            t = 1 - step * dt
            xt = self.reverse(Unet=Unet,x_hat=x_hat,xt=xt,t=t, dt=dt)
        return xt
    
    def 