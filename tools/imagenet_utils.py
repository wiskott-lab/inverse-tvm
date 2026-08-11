from pathlib import Path

from torchvision import datasets


def get_imagenet_split(root, split, transform=None):
    root = Path(root)
    split_dir = root / split
    if split_dir.is_dir():
        return datasets.ImageFolder(split_dir, transform=transform)
    return datasets.ImageNet(root, split=split, transform=transform)
