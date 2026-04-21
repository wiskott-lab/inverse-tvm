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

import tools.swin_utils as su
import numpy as np
import wandb
import tools.vit_utils as vu


@torch.no_grad()
def test_inverse_swin(models, dataloader, model, run=None):
    num_inputs = 0
    sum_losses = np.zeros(shape=(len(models),))
    with torch.no_grad():
        for batch_id, (img, target) in enumerate(dataloader):
            img, target = img.to(config.DEVICE), target.to(config.DEVICE)
            embs = su.tensor_to_embs(tensor=img, model=model)
            recon_embs = su.invert_embs(embs=embs, inv_networks=models)
            for i in range(len(recon_embs)):
                if i == 0:
                    loss = F.mse_loss(input=cu.normalize(recon_embs[i]), target=embs[i]).item()
                else:
                    loss = F.mse_loss(input=recon_embs[i], target=embs[i]).item()
                sum_losses[i] += loss * len(img)
            num_inputs += len(img)
        sum_losses = sum_losses / num_inputs
        if run:
            for i in range(len(models)):
                run[f"val/{i}"].append(sum_losses[i])
    return sum_losses

@torch.no_grad()
def test_classic_in_parallel(all_models, dataloader, swin, step=None):
    sum_losses, num_inputs = torch.zeros(len(all_models)).to(config.DEVICE), 0
    for batch_id, (img, _) in enumerate(dataloader):
        img = img.to(config.DEVICE)
        int_embs = su.tensor_to_embs(tensor=img, model=swin)
        for i in range(len(all_models)):
            current_model = all_models[i]
            current_emb = int_embs[i + 1]
            for j in range(len(current_model)):
                current_emb = current_model[j](current_emb)
            sum_losses[i] += F.mse_loss(input=cu.normalize(current_emb), target=img) * img.shape[0]
        num_inputs += img.shape[0]
    sum_losses /= num_inputs
    for i in range(len(all_models)):
        wandb.log({"val/step": step, f'val/loss_{i}': sum_losses[i]})
    return sum_losses

@torch.no_grad()
def test_classic_in_par(vit, bb_to_img, enc_to_img, dataloader, run=None, normalize=False):
    sum_bb_loss, sum_enc_loss,  num_inputs = 0, 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        if normalize:
            denormalized_img = cu.denormalize(img)
        else:
            denormalized_img = img
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)

        if bb_to_img is not None:
            bb_to_img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=bb_to_img[0])
            bb_loss = F.mse_loss(input=bb_to_img_recon, target=denormalized_img)
            sum_bb_loss += bb_loss * len(img)
        if enc_to_img is not None:
            enc_to_img_recon = su.enc_emb_to_tensor(enc_emb=enc_emb, inv_bb=enc_to_img[0], inv_enc=enc_to_img[1])
            enc_loss = F.mse_loss(input=enc_to_img_recon, target=denormalized_img)
            sum_enc_loss += enc_loss * len(img)
        num_inputs += len(img)
    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/enc_loss"].append((sum_enc_loss / num_inputs).item())
    return (sum_bb_loss / num_inputs).item(), (sum_enc_loss / num_inputs).item()

@torch.no_grad()
def test_inverse_swin_end(models, dataloader, swin, run=None):
    num_inputs = 0
    sum_loss = 0
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE), target.to(config.DEVICE)
        emb = su.tensor_to_embs(tensor=img, model=swin)[-2]
        recon = su.chain_invert(emb=emb, inv_networks=models)
        loss = F.mse_loss(input=cu.normalize(recon), target=img)
        sum_loss += loss * len(img)
        num_inputs += len(img)
    sum_loss = (sum_loss / num_inputs).item()
    if run:
        run[f"val/recon"].append(sum_loss)
    return sum_loss
