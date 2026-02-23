import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from auto_encoder.dataset import BBoxDataset
from auto_encoder.models import Autoencoder_v1, Autoencoder_VAE_v1
from constants import IMAGE_DIR_PATH

MODEL_DIR = Path("models")


def main(model_name: str, latent_dim: int, batch_size: int = 64):
    DEVICE = "cpu"

    MODEL_PATH = MODEL_DIR / model_name
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    OUTPUT_DIR = Path("latent_assets")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Dataset
    dataset = BBoxDataset(
        image_dir=IMAGE_DIR_PATH,
        annotations_path=Path("labels/auto_annotations_fire_watcher-1.json"),
        max_items=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model: infer type from filename
    if "vae" in MODEL_PATH.stem:
        model = Autoencoder_VAE_v1(latent_dim=latent_dim)
    else:
        model = Autoencoder_v1(latent_dim=latent_dim)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Compute latents
    latents = []
    images = []
    with torch.no_grad():
        logger.info("Compute latent vectors")
        for imgs in loader:
            imgs = imgs.to(DEVICE)
            z = model.encoder(imgs)
            if isinstance(z, tuple):
                z = z[0]  # fallback for tuple output
            latents.append(z.cpu())
            images.append(imgs.cpu())

    latents = torch.cat(latents).numpy()
    images = torch.cat(images).numpy()

    # t-SNE
    logger.info("Compute t-SNE embeddings")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    latents_2d = tsne.fit_transform(latents)

    # Save assets
    logger.info("Save latent assets")
    model_name_tag = MODEL_PATH.stem
    np.save(OUTPUT_DIR / f"{model_name_tag}_latents.npy", latents)
    np.save(OUTPUT_DIR / f"{model_name_tag}_tsne_2d.npy", latents_2d)
    np.save(OUTPUT_DIR / f"{model_name_tag}_images.npy", images)

    # # Clustering
    # logger.info("Prepare default clustering (k=5)")
    # kmeans = KMeans(n_clusters=5, random_state=42).fit(latents)
    # np.save(OUTPUT_DIR / f"{model_name_tag}_clusters.npy", kmeans.labels_)

    logger.success(f"Latent assets saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute latents, t-SNE, and clustering for an autoencoder/VAE."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Filename of the trained model in /models (e.g., autoencoder_2k.pth)",
    )
    parser.add_argument(
        "--latent_dim", type=int, help="Latent dimensionality of the model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference (default=64)",
    )

    args = parser.parse_args()
    main(args.model_name, args.latent_dim, args.batch_size)
