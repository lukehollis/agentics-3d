import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_size=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(1024, latent_size)
        self.fc_var = nn.Linear(1024, latent_size)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_size, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 6, stride=2),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 1024, 1, 1)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class MDN_RNN(nn.Module):
    def __init__(self, latent_size=32, hidden_size=256, n_gaussians=5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_gaussians = n_gaussians
        
        self.lstm = nn.LSTM(latent_size + 1, hidden_size, batch_first=True)  # +1 for action
        
        # MDN outputs
        self.z_pi = nn.Linear(hidden_size, n_gaussians)
        self.z_mu = nn.Linear(hidden_size, n_gaussians * latent_size)
        self.z_sigma = nn.Linear(hidden_size, n_gaussians * latent_size)
        
    def forward(self, x, actions, hidden=None):
        # Combine latent and action
        combined = torch.cat([x, actions.unsqueeze(-1)], dim=-1)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(combined, hidden)
        
        # Get mixture params
        pi = F.softmax(self.z_pi(lstm_out), dim=-1)
        mu = self.z_mu(lstm_out).view(-1, self.n_gaussians, self.latent_size)
        sigma = torch.exp(self.z_sigma(lstm_out)).view(-1, self.n_gaussians, self.latent_size)
        
        return pi, mu, sigma, hidden

def train_world_model(train_dataset, val_dataset, device='cuda', epochs=100):
    # Initialize models
    vae = VAE().to(device)
    mdn_rnn = MDN_RNN().to(device)
    
    # Setup training
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    rnn_optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=1e-4)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        vae.train()
        mdn_rnn.train()
        
        for batch in train_loader:
            # VAE forward pass
            obs = batch['observations'].to(device)
            recon, mu, log_var = vae(obs)
            
            # VAE loss
            recon_loss = F.mse_loss(recon, obs)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = recon_loss + 0.0001 * kl_loss
            
            # VAE backward pass
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()
            
            # MDN-RNN forward pass
            with torch.no_grad():
                z, _ = vae.encode(obs)
            
            actions = batch['actions'].to(device)
            pi, mu, sigma, _ = mdn_rnn(z, actions)
            
            # MDN-RNN loss
            target = z[1:]  # Next latent states
            mdn_loss = mdn_gaussian_loss(target, pi[:-1], mu[:-1], sigma[:-1])
            
            # MDN-RNN backward pass
            rnn_optimizer.zero_grad()
            mdn_loss.backward()
            rnn_optimizer.step()
            
        # Validation step
        validate_models(vae, mdn_rnn, val_dataset, device)
        
    return vae, mdn_rnn

def mdn_gaussian_loss(y, pi, mu, sigma):
    m = torch.distributions.Normal(mu, sigma)
    loss = torch.exp(m.log_prob(y.unsqueeze(1).expand_as(mu)))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss + 1e-6)
    return torch.mean(loss)
