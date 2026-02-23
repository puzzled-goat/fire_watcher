import warnings

import matplotlib.pyplot as plt

from auto_encoder.dataset import BBoxDataset

warnings.filterwarnings("ignore", category=UserWarning)
import torch


def show_reconstruction_from_dataset(
    model,
    dataset: BBoxDataset,
    file_name: str,
    device="cpu",
):
    if file_name not in dataset.index_by_name:
        raise ValueError(f"{file_name} not found in dataset")

    model.eval()
    idx = dataset.index_by_name[file_name]

    img = dataset[idx].unsqueeze(0).to(device)

    with torch.no_grad():
        recon = model(img)

    orig_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(orig_np)
    axs[0].set_title("Original bbox")
    axs[0].axis("off")

    axs[1].imshow(recon_np)
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
