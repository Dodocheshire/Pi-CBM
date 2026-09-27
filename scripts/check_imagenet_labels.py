"""Check a sampled ImageNet directory against pretrained ResNet-50 labels.

Run this before training: an ImageNet-shaped download can contain randomized
labels while still having exactly 1,000 class folders and valid image files.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


class LabeledImages(Dataset):
    def __init__(self, root: Path, class_index: Path, limit: int):
        mapping = json.loads(class_index.read_text())
        labels = {entry[0]: int(index) for index, entry in mapping.items()}
        files = sorted(root.glob("*/*.JPEG"))
        self.files = random.Random(42).sample(files, min(limit, len(files)))
        self.labels = labels
        self.transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, self.labels[path.parent.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--class-index", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    data = LabeledImages(args.images, args.class_index, args.limit)
    loader = DataLoader(data, batch_size=64, num_workers=4)
    top1 = top5 = 0
    with torch.inference_mode():
        for images, labels in loader:
            guesses = model(images.to(device)).topk(5, dim=1).indices.cpu()
            top1 += (guesses[:, 0] == labels).sum().item()
            top5 += (guesses == labels[:, None]).any(dim=1).sum().item()
    print(json.dumps({"images": len(data), "top1": top1 / len(data), "top5": top5 / len(data)}))


if __name__ == "__main__":
    main()
