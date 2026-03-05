import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3*2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            CNNBlock(64, 128, kernel_size=4, stride=2, padding=1),
            CNNBlock(128, 256, kernel_size=4, stride=2, padding=1),
            CNNBlock(256, 512, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )


    def forward(self, x, y):
        X = torch.cat([x, y], dim=1)
        return self.model(X)

class Block_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block_Down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
         return self.conv(x)

class Block_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block_Up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
         return self.conv(x)
    
    
class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()
      self.enc1 = Block_Down(3, 64)
      self.enc2 = Block_Down(64, 128)
      self.enc3 = Block_Down(128, 256)
      self.enc4 = Block_Down(256, 512)

        
      self.dec1 = Block_Up(512, 256)  
      self.dec2 = Block_Up(512, 128)  
      self.dec3 = Block_Up(256, 64)                
      self.dec4 = nn.ConvTranspose2d(128, 3, 4, 2, 1)  
      self.tanh = nn.Tanh()

    def forward(self, x):
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)


        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        out = self.dec4(d3)
        out = self.tanh(out)

        return out








