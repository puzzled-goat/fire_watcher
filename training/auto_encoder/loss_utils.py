import torch
import torch.nn.functional as F


def ae_loss(recon_x, x, **_):
    """
    Standard autoencoder reconstruction loss.
    """
    return F.mse_loss(recon_x, x)


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """
    VAE loss = reconstruction + beta * KL divergence.
    """
    mse = F.mse_loss(recon_x, x)
    kld_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld_sum / x.numel()
    return mse + beta * kld
