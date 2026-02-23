from pathlib import Path
from typing import Callable, Type

import torch
import torch.optim as optim
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from auto_encoder.dataset import BBoxDataset
from auto_encoder.loss_utils import ae_loss
from auto_encoder.models import Autoencoder_v1


def train_autoencoder(
    image_dir: Path,
    annotations_path: Path,
    model_cls: Type[nn.Module] = Autoencoder_v1,
    loss_fn: Callable = ae_loss,
    max_images: int = 500,
    epochs: int = 10,
    batch_size: int = 8,
    latent_dim: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    This function is model-agnostic and supports both standard autoencoders and
    variational autoencoders by accepting the model class and loss function as
    parameters. The loss function is expected to match the outputs of the model’s
    forward method (e.g., reconstruction only, or reconstruction plus latent
    statistics).

    Args:
        image_dir (Path): Directory containing input images.
        annotations_path (Path): Path to the JSON file with bounding box annotations.
        model_cls (Type[nn.Module]): Autoencoder model class to instantiate.
        loss_fn (Callable): Loss function used for training.
        max_images (int): Maximum number of images to load from the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        latent_dim (int): Latent dimensionality passed to the model constructor.
        lr (float): Learning rate for the optimizer.
        device (str): Device identifier (e.g., "cpu" or "cuda").

    Returns:
        tuple[nn.Module, Dataset]: The trained model and the dataset used for training.

    Placeholder: Extend to support validation loops, learning-rate scheduling,
    or metric logging beyond the training loss.
    """

    dataset = BBoxDataset(
        image_dir=image_dir,
        annotations_path=annotations_path,
        max_items=max_images,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # important for low-RAM CPU
        pin_memory=False,
    )

    model = model_cls(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info(
        f"Training {model_cls.__name__} "
        f"on {len(dataset)} images "
        f"for {epochs} epochs"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs in dataloader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            if loss_fn.__name__ == "ae_loss":
                loss = loss_fn(outputs, imgs)
            elif loss_fn.__name__ == "vae_loss":
                loss = loss_fn(outputs, imgs, model.mu, model.logvar)
            else:
                raise ValueError("Unsupported loss function")

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    model_path = f"{model_cls.__name__}_latent{latent_dim}.pth"
    torch.save(model.state_dict(), model_path)

    logger.success(f"Training complete. Model saved to {model_path}")
    return model, dataset
