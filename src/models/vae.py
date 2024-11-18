import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_size=32, base_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 3, stride=1, padding=1),
            ResidualBlock(base_channels),
            nn.MaxPool2d(2, 2),  # 32x32
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1),
            ResidualBlock(base_channels * 2),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=1, padding=1),
            ResidualBlock(base_channels * 4),
            nn.MaxPool2d(2, 2),  # 8x8
            
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=1, padding=1),
            ResidualBlock(base_channels * 8),
            nn.MaxPool2d(2, 2),  # 4x4
        )
        
        # Latent space
        self.fc_mu = nn.Linear(base_channels * 8 * 4 * 4, latent_size)
        self.fc_logvar = nn.Linear(base_channels * 8 * 4 * 4, latent_size)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_size, base_channels * 8 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            ResidualBlock(base_channels * 4),
            
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            ResidualBlock(base_channels * 2),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            ResidualBlock(base_channels),
            
            nn.ConvTranspose2d(base_channels, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4)
        return self.decoder(x)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar