from torch.functional import F
from tools import detr_utils
import tools.coco_utils as cu
import torch
from torch.functional import F
import config
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import  RandomResizedCropAndInterpolation
from PIL import ImageFilter, ImageOps
from torchvision import  transforms
import random
from torchvision.transforms import InterpolationMode
import math
import torch.distributed as dist
import sys
import tools.resnet_utils as ru
import numpy as np

@torch.no_grad()
def test_inverse_models(inverse_models, dataloader, forward_model, input_proj=None, run=None):
    num_inputs = 0
    sum_losses = np.zeros(shape=(len(inverse_models),))
    with torch.no_grad():
        for model in inverse_models:
            model.eval()
        for batch_id, (nested_tensor, _) in enumerate(dataloader):
            img = nested_tensor.tensors.to(config.DEVICE)
            embs = ru.tensor_to_embs(tensor=img, model=forward_model, input_proj=input_proj)
            recon_embs = ru.invert_embs(embs=embs, inv_networks=inverse_models)
            for i in range(len(recon_embs)):
                if i == 0:
                    loss = F.mse_loss(input=cu.normalize(recon_embs[i]), target=embs[i]).item()
                else:
                    loss = F.mse_loss(input=recon_embs[i], target=embs[i]).item()
                sum_losses[i] += loss * len(img)
            num_inputs += len(img)
        sum_losses = sum_losses / num_inputs
        if run:
            for i in range(len(inverse_models)):
                run[f"val/{i}"].append(sum_losses[i])
    return sum_losses


@torch.no_grad()
def test_inverse_models_end(models, dataloader, model, run=None):
    num_inputs = 0
    sum_loss = 0
    for model in models:
        model.eval()
    for batch_id, (nested_tensor, _) in enumerate(dataloader):
        img = nested_tensor.tesnors.to(config.DEVICE)
        emb = ru.tensor_to_embs(tensor=img, model=model)[-2]
        recon = ru.chain_invert(emb=emb, inv_networks=models)
        loss = F.mse_loss(input=cu.normalize(recon), target=img)
        sum_loss += loss * len(img)
        num_inputs += len(img)
    sum_loss = (sum_loss / num_inputs).item()
    if run:
        run[f"val/recon"].append(sum_loss)
    return sum_loss
