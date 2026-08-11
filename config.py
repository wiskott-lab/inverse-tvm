from pathlib import Path
import torch

EXPERIMENT_NAME = "inverse-tvm"
PROJECT = EXPERIMENT_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMGNET1k_DIR = Path("your/imgnet1k/dir")
COCO_DIR = Path("your/coco/dataset/dir")
IMGNET1k_TRAIN_SPLIT = "train"
IMGNET1k_VAL_SPLIT = "val"
RUNS_DIR = Path("runs")
TMP_DIR = Path("tmp")
