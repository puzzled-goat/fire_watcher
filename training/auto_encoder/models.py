import torch
from loguru import logger
from torch import nn


class Autoencoder_v1(nn.Module):
    """
    Convolutional autoencoder with a fully connected bottleneck.

    Encodes 128x128 RGB images into a fixed-size latent vector using strided
    convolutions followed by flattening and a linear projection.
    Decodes the latent vector back to image space using linear expansion and transposed convolutions.

    model = Autoencoder()
    summary(model, input_size=(32, 3 , 128 , 128))
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Autoencoder_v1                           [32, 3, 128, 128]         --
    ├─Sequential: 1-1                        [32, 64]                  --
    │    └─Conv2d: 2-1                       [32, 16, 64, 64]          448
    │    └─ReLU: 2-2                         [32, 16, 64, 64]          --
    │    └─Conv2d: 2-3                       [32, 32, 32, 32]          4,640
    │    └─ReLU: 2-4                         [32, 32, 32, 32]          --
    │    └─Conv2d: 2-5                       [32, 64, 16, 16]          18,496
    │    └─ReLU: 2-6                         [32, 64, 16, 16]          --
    │    └─Flatten: 2-7                      [32, 16384]               --
    │    └─Linear: 2-8                       [32, 64]                  1,048,640
    ├─Sequential: 1-2                        [32, 3, 128, 128]         --
    │    └─Linear: 2-9                       [32, 16384]               1,064,960
    │    └─Unflatten: 2-10                   [32, 64, 16, 16]          --
    │    └─ConvTranspose2d: 2-11             [32, 32, 32, 32]          18,464
    │    └─ReLU: 2-12                        [32, 32, 32, 32]          --
    │    └─ConvTranspose2d: 2-13             [32, 16, 64, 64]          4,624
    │    └─ReLU: 2-14                        [32, 16, 64, 64]          --
    │    └─ConvTranspose2d: 2-15             [32, 3, 128, 128]         435
    │    └─Sigmoid: 2-16                     [32, 3, 128, 128]         --
    ==========================================================================================
    Total params: 2,160,707
    Trainable params: 2,160,707
    Non-trainable params: 0
    Total mult-adds (Units.GIGABYTES): 1.87
    ==========================================================================================
    Input size (MB): 6.29
    Forward/backward pass size (MB): 71.32
    Params size (MB): 8.64
    Estimated Total Size (MB): 86.25
    ==========================================================================================
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        # Input: 3 x 128 x 128 = 49,152
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class Autoencoder_v2(nn.Module):
    """
    Fully convolutional autoencoder with a spatial bottleneck.

    Encodes 128x128 RGB images into a compact latent representation using only
    convolutions, ending with a 1x1 bottleneck and global average pooling.
    Decoding is performed with transposed convolutions to restore spatial resolution.

    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Autoencoder_v2                           [32, 3, 128, 128]         --
    ├─Sequential: 1-1                        [32, 64, 1, 1]            --
    │    └─Conv2d: 2-1                       [32, 16, 64, 64]          448
    │    └─ReLU: 2-2                         [32, 16, 64, 64]          --
    │    └─Conv2d: 2-3                       [32, 32, 32, 32]          4,640
    │    └─ReLU: 2-4                         [32, 32, 32, 32]          --
    │    └─Conv2d: 2-5                       [32, 64, 16, 16]          18,496
    │    └─ReLU: 2-6                         [32, 64, 16, 16]          --
    │    └─Conv2d: 2-7                       [32, 128, 8, 8]           73,856
    │    └─ReLU: 2-8                         [32, 128, 8, 8]           --
    │    └─Conv2d: 2-9                       [32, 64, 8, 8]            8,256
    │    └─AdaptiveAvgPool2d: 2-10           [32, 64, 1, 1]            --
    ├─Sequential: 1-2                        [32, 3, 128, 128]         --
    │    └─ConvTranspose2d: 2-11             [32, 128, 8, 8]           524,416
    │    └─ReLU: 2-12                        [32, 128, 8, 8]           --
    │    └─ConvTranspose2d: 2-13             [32, 64, 16, 16]          131,136
    │    └─ReLU: 2-14                        [32, 64, 16, 16]          --
    │    └─ConvTranspose2d: 2-15             [32, 32, 32, 32]          32,800
    │    └─ReLU: 2-16                        [32, 32, 32, 32]          --
    │    └─ConvTranspose2d: 2-17             [32, 16, 64, 64]          8,208
    │    └─ReLU: 2-18                        [32, 16, 64, 64]          --
    │    └─ConvTranspose2d: 2-19             [32, 3, 128, 128]         771
    │    └─Sigmoid: 2-20                     [32, 3, 128, 128]         --
    ==========================================================================================
    Total params: 803,027
    Trainable params: 803,027
    Non-trainable params: 0
    Total mult-adds (Units.GIGABYTES): 5.23
    ==========================================================================================
    Input size (MB): 6.29
    Forward/backward pass size (MB): 76.55
    Params size (MB): 3.21
    Estimated Total Size (MB): 86.05
    ==========================================================================================
    """

    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Bottleneck
            nn.Conv2d(128, latent_dim, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class Autoencoder_VAE_v1(nn.Module):
    """
    Variational Autoencoder (VAE) with an Autoencoder_v1-like structure.

    Source: https://github.com/pmarinroig/c-vae/blob/main/pytorch_poc

    IMPORTANT:
    - This model is intentionally made compatible with the existing
      `train_autoencoder` function, which:
        * Expects `forward(x)` to return ONLY the reconstructed image
        * Uses MSE loss externally
        * Is unaware of KL-divergence

    Design choices for compatibility:
    - Encoder outputs concatenated [mu, logvar]
    - Reparameterization is done internally
    - Forward returns ONLY the reconstruction
    - KL loss is computed and stored on `self.kl_loss`
      (available if you later want to extend the training loop)

    This preserves VAE behavior without breaking the current training code.
    """

    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        # Placeholder to expose KL loss if needed later
        self.kl_loss = torch.tensor(0.0)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)

        # store mu/logvar on the model for external access
        self.mu = mu
        self.logvar = logvar

        z = self.reparameterize(mu, logvar)

        # Store KL loss for optional future use
        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        out = self.decoder(z)
        return out
