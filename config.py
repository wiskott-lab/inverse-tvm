import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMGNET1k_DIR = Path("your/imgnet_1k_dir")
COCO_DIR = Path("your/coco_dataset_dir")
