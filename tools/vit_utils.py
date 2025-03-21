import torch
import timm
from fontTools.unicodedata import block

from modules.inv_vit_bb import models as inv_bb_module
from modules.inv_vit_encoder import models as inv_enc_module
from tools.neptune_utils import init_model_from_neptune


def tensor_to_bb_emb(tensor, vit):
    x = vit.patch_embed(tensor)
    x = vit._pos_embed(x)
    return x

def tensor_to_enc_emb(tensor, vit):
    bb_emb = tensor_to_bb_emb(tensor, vit)
    enc_emb = bb_emb_to_enc_emb(bb_emb, vit)
    return enc_emb

def tensor_to_vit_out(tensor, vit):
    enc_emb = tensor_to_enc_emb(tensor, vit)
    vit_out = enc_emb_to_vit_out(enc_emb, vit)
    return vit_out


def bb_emb_to_enc_emb(bb_emb, vit):
    enc_emb = vit.blocks(bb_emb)
    return enc_emb

def bb_emb_to_vit_out(input_emb, vit):
    enc_emb = bb_emb_to_enc_emb(input_emb, vit)
    vit_out = enc_emb_to_vit_out(enc_emb, vit)
    return vit_out


def bb_emb_to_tensor(bb_emb, inv_bb):
    tensor = inv_bb(bb_emb)
    return tensor


def enc_emb_to_vit_out(enc_emb, vit):
    vit_out = vit.forward_head(enc_emb)
    return vit_out


def enc_emb_to_bb_emb(enc_emb, inv_enc):
    bb_emb = inv_enc(enc_emb)
    return bb_emb


def enc_emb_to_tensor(enc_emb, inv_bb, inv_enc):
    bb_emb = enc_emb_to_bb_emb(enc_emb, inv_enc)
    tensor = bb_emb_to_tensor(bb_emb, inv_bb)
    return tensor

def int_enc_rep_to_int_enc_rep(int_enc_rep, vit, from_layer, to_layer):
    for i in range(from_layer, to_layer):
        enc_block = vit.blocks[i]
        int_enc_rep = int_enc_rep + enc_block.drop_path1(enc_block.ls1(enc_block.attn(enc_block.img_cross_norm(int_enc_rep))))
        int_enc_rep = int_enc_rep + enc_block.drop_path2(enc_block.ls2(enc_block.mlp(enc_block.reg_cross_norm(int_enc_rep))))
    return int_enc_rep


def get_int_reps_from_bb_emb(x, vit):
    int_reps_1 = [x]
    int_reps_2 = [x]
    for block in vit.blocks:
        x = x + block.drop_path1(block.ls1(block.attn(block.img_cross_norm(x))))
        int_reps_1.append(x)
        x = x + block.drop_path2(block.ls2(block.mlp(block.reg_cross_norm(x))))
        int_reps_2.append(x)
    return int_reps_1, int_reps_2


def get_emb_recons_from_tensor(tensor, vit, inv_bb, inv_enc):
    bb_emb = tensor_to_bb_emb(tensor, vit)
    enc_emb = bb_emb_to_enc_emb(bb_emb, vit)
    bb_emb_recon = bb_emb_to_tensor(bb_emb, inv_bb)
    enc_emb_recon = enc_emb_to_tensor(enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
    return bb_emb_recon, enc_emb_recon



def get_int_reps_from_tensor(tensor, vit):
    x = tensor_to_bb_emb(tensor, vit)
    int_reps_1 = [x]
    int_reps_2 = [x]
    for block in vit.blocks:
        x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
        int_reps_1.append(x)
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        int_reps_2.append(x)
    return int_reps_1, int_reps_2


# Misc
def init_vit_modules(vit_id=None, inv_bb_id=None, inv_enc_id=None):
    vit, inv_bb, inv_enc = None, None, None
    if vit_id is not None:
        vit = timm.create_model(vit_id, pretrained=True)
    if inv_bb_id is not None:
        inv_bb = init_model_from_neptune(run_id=inv_bb_id, module=inv_bb_module)
    if inv_enc_id is not None:
        inv_enc = init_model_from_neptune(run_id=inv_enc_id, module=inv_enc_module)
    return vit, inv_bb, inv_enc
