from pathlib import Path
import torch

PROJECT = 'your-project-name'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMGNET1k_DIR = Path("your/imgnet1k/dir")
COCO_DIR = Path("your/coco/dataset/dir")

