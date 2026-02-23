import json
from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class BBoxDataset(Dataset):
    """
    PyTorch Dataset for images with bounding box annotations.

    This dataset loads images, crops them according to bounding box annotations,
    resizes the cropped region, and converts it to a tensor.
    """

    def __init__(
        self,
        image_dir: Path,
        annotations_path: Path,
        output_size=(128, 128),
        max_items: int | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            image_dir (Path): Directory containing image files.
            annotations_path (Path): Path to a JSON file with annotations.
                format: {"screenshot_2026-01-18_18-11-01.jpg": {
                            "labeled": false,
                            "bbox": [258.0, 174.0, 155.0, 147.0]}
            output_size (tuple[int, int]): Target (width, height) for resized crops.
            max_items (int | None): Optional limit on number of samples loaded.

        Placeholder: Document JSON annotation file structure in more detail.
        """
        self.image_dir = image_dir
        self.output_size = output_size

        self.annotations = json.loads(annotations_path.read_text())

        self.file_names = [
            fname for fname, data in self.annotations.items() if data.get("bbox")
        ]

        if max_items is not None:
            self.file_names = self.file_names[:max_items]

        self.index_by_name = {name: i for i, name in enumerate(self.file_names)}

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = self.image_dir / file_name

        img = Image.open(img_path).convert("RGB")
        x, y, w, h = self.annotations[file_name]["bbox"]

        cropped = img.crop((x, y, x + w, y + h))
        cropped = cropped.resize(self.output_size)

        return self.transform(cropped)
