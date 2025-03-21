import torch
from tools.neptune_utils import init_model_from_neptune
from modules.detr.hubconf import detr_resnet50
from modules.inverse_backbone import models as inv_bb_module
from modules.inverse_encoder import models as inv_enc_module
from modules.inverse_decoder import models as inv_dec_module
from modules.detr import models as detr_module


# from nested tensor
# forward
def nested_tensor_to_bb_emb(nested_tensor, detr):
    features, pos = detr.backbone(nested_tensor)
    src, mask = features[-1].decompose()
    bb_emb = detr.input_proj(src).flatten(2).permute(2, 0, 1)
    return bb_emb, pos[-1].flatten(2).permute(2, 0, 1), mask.flatten(1)


def nested_tensor_to_enc_emb(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor, detr)
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    return enc_emb, pos, mask


def nested_tensor_to_dec_emb(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor, detr)
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    return dec_emb, pos, mask


def nested_tensor_to_detr_out(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor, detr)
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    detr_out = dec_emb_to_detr_out(dec_emb, detr)
    return detr_out, pos, mask


# from backbone embedding
# forward
def bb_emb_to_enc_emb(bb_emb, detr, mask, pos):
    enc_emb = detr.transformer.encoder(src=bb_emb, src_key_padding_mask=mask, pos=pos)
    return enc_emb


def bb_emb_to_dec_emb(bb_emb, detr, mask, pos):
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    return dec_emb


def bb_emb_to_detr_out(bb_emb, detr, mask, pos):
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    detr_out = dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
    return detr_out


# backward
def bb_emb_to_img_tensor(bb_emb, inv_bb, w_max_32=20, h_max_32=15):
    spatial_rep = sequence_to_spatial(bb_emb, w_max_32=w_max_32, h_max_32=h_max_32)
    img = inv_bb(spatial_rep)
    return img


# from encoder embedding
# forward
def enc_emb_to_dec_emb(enc_emb, detr, mask, pos):
    query_embed = detr.query_embed.weight
    query_embed = query_embed.unsqueeze(1).repeat(1, enc_emb.shape[1], 1)
    tgt = torch.zeros_like(query_embed)
    dec_emb = detr.transformer.decoder(tgt, enc_emb, memory_key_padding_mask=mask, pos=pos, query_pos=query_embed)
    return dec_emb


def enc_emb_to_detr_out(enc_emb, detr, mask, pos):
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    detr_out = dec_emb_to_detr_out(dec_emb, detr)
    return detr_out


# backward
def enc_emb_to_bb_emb(enc_emb, inv_enc, mask, pos):
    bb_emb = inv_enc(enc_emb, src_key_padding_mask=mask, pos=pos)
    return bb_emb


def enc_emb_to_img_tensor(enc_emb, inv_enc, inv_bb, mask, pos, w_max_32=20, h_max_32=15):
    bb_emb = enc_emb_to_bb_emb(enc_emb, inv_enc, mask, pos)
    img = bb_emb_to_img_tensor(bb_emb, inv_bb, w_max_32=w_max_32, h_max_32=h_max_32)
    return img


# from decoder embedding
# forward
def dec_emb_to_detr_out(dec_emb, detr):
    dec_emb = dec_emb.transpose(1, 2)
    outputs_class = detr.class_embed(dec_emb)
    outputs_coord = detr.bbox_embed(dec_emb).sigmoid()
    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    return out


# backward
def dec_emb_to_enc_emb(dec_emb, pos, inv_dec):
    dec_emb = dec_emb[-1]
    dec_pos = inv_dec.query_embed.weight.unsqueeze(1).repeat(1, dec_emb.shape[1], 1)
    tgt = torch.zeros_like(pos)
    enc_emb = inv_dec(tgt, memory=dec_emb, pos=dec_pos, query_pos=pos)[0]
    return enc_emb


def dec_emb_to_bb_emb(dec_emb, pos, mask, inv_dec, inv_enc):
    enc_emb = dec_emb_to_enc_emb(dec_emb=dec_emb, pos=pos, inv_dec=inv_dec)
    bb_emb = enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, pos=pos, mask=mask)
    return bb_emb


def dec_emb_to_img_tensor(dec_emb, mask, pos, inv_dec, inv_enc, inv_bb, w_max_32=20, h_max_32=15):
    enc_emb = dec_emb_to_enc_emb(dec_emb, pos, inv_dec)
    bb_emb = enc_emb_to_bb_emb(enc_emb=enc_emb, pos=pos, mask=mask, inv_enc=inv_enc)
    img_tensor = bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb, w_max_32=w_max_32, h_max_32=h_max_32)
    return img_tensor


# intermediate representations
def enc_emb_to_dec_reps(enc_emb, detr, mask, pos):
    query_embed = detr.query_embed.weight
    query_embed = query_embed.unsqueeze(1).repeat(1, enc_emb.shape[1], 1)
    tgt = torch.zeros_like(query_embed)
    hs = detr.transformer.decoder(tgt, enc_emb, memory_key_padding_mask=mask, pos=pos, query_pos=query_embed)
    int_reps = list(torch.unbind(hs, dim=0))
    int_reps.insert(0, tgt)
    return int_reps


def bb_emb_to_enc_reps(bb_emb, detr, mask, pos):
    int_reps = [bb_emb]
    for layer in detr.transformer.encoder.layers:
        int_reps.append(layer(int_reps[-1], src_key_padding_mask=mask, pos=pos))
    return int_reps


@torch.no_grad()
def get_int_reps_from_nested_tensor(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
    int_enc_reps = bb_emb_to_enc_reps(bb_emb=bb_emb, detr=detr, mask=mask, pos=pos)
    int_dec_reps = enc_emb_to_dec_reps(enc_emb=int_enc_reps[-1], detr=detr, mask=mask, pos=pos)
    return int_enc_reps, int_dec_reps, pos, mask


@torch.no_grad()
def get_embs_from_nested_tensor(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
    enc_emb = bb_emb_to_enc_emb(bb_emb=bb_emb, detr=detr, pos=pos, mask=mask)
    dec_emb = enc_emb_to_dec_emb(enc_emb=enc_emb, pos=pos, mask=mask, detr=detr)
    return bb_emb, enc_emb, dec_emb, mask, pos


@torch.no_grad()
def get_emb_recons_from_nested_tensor(nested_tensor, detr, inv_bb=None, inv_enc=None, inv_dec=None):
    bb_emb, enc_emb, dec_emb, mask, pos = get_embs_from_nested_tensor(nested_tensor=nested_tensor, detr=detr)
    recon_from_bb_emb, recon_from_enc_emb, recon_from_dec_emb = get_imgs_from_embs(bb_emb=bb_emb, enc_emb=enc_emb,
                                                                                   dec_emb=dec_emb, mask=mask,
                                                                                   pos=pos, inv_bb=inv_bb,
                                                                                   inv_enc=inv_enc, inv_dec=inv_dec)
    return recon_from_bb_emb, recon_from_enc_emb, recon_from_dec_emb


@torch.no_grad()
def get_imgs_from_embs(bb_emb=None, enc_emb=None, dec_emb=None, mask=None, pos=None, inv_bb=None, inv_enc=None,
                       inv_dec=None):
    recon_bb_emb, recon_enc_emb, recon_dec_emb = None, None, None
    if inv_bb is not None:
        recon_bb_emb = bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
    if inv_enc is not None:
        recon_enc_emb = enc_emb_to_img_tensor(enc_emb=enc_emb, inv_bb=inv_bb, pos=pos, mask=mask, inv_enc=inv_enc)
    if inv_dec is not None:
        recon_dec_emb = dec_emb_to_img_tensor(dec_emb=dec_emb, inv_bb=inv_bb, pos=pos, mask=mask, inv_enc=inv_enc,
                                              inv_dec=inv_dec)
    return recon_bb_emb, recon_enc_emb, recon_dec_emb


@torch.no_grad()
def get_imgs_from_int_reps(int_enc_reps, int_dec_reps, pos, mask, inv_bb, inv_enc, inv_dec):
    int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb = [], [], []
    # reconstructions from intermediate encoder representations used as backbone embedding
    # reconstructions from intermediate encoder representations used as encoder embedding
    # reconstructions from intermediate decoder representations used as decoder embedding
    for rep in int_dec_reps:
        recon = dec_emb_to_img_tensor(dec_emb=rep.unsqueeze(0), mask=mask, inv_bb=inv_bb, inv_dec=inv_dec,
                                      inv_enc=inv_enc, pos=pos)
        int_recons_from_dec_emb.append(recon)
    for rep in int_enc_reps:
        recon_bb_emb = bb_emb_to_img_tensor(bb_emb=rep, inv_bb=inv_bb)
        int_recons_from_bb_emb.append(recon_bb_emb)
        recon_enc_emb = enc_emb_to_img_tensor(enc_emb=rep, inv_enc=inv_enc, inv_bb=inv_bb, pos=pos, mask=mask)
        int_recons_from_enc_emb.append(recon_enc_emb)
    return int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb


@torch.no_grad()
def get_int_recons_from_nested_tensor(nested_tensor, detr, inv_bb, inv_enc, inv_dec):
    int_enc_reps, int_dec_reps, pos, mask = get_int_reps_from_nested_tensor(nested_tensor, detr)
    int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb = \
        get_imgs_from_int_reps(int_enc_reps=int_enc_reps, int_dec_reps=int_dec_reps, pos=pos, mask=mask,
                               inv_bb=inv_bb, inv_enc=inv_enc, inv_dec=inv_dec)
    return int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb


# Misc
def init_detr_modules(detr_id=None, inv_bb_id=None, inv_enc_id=None, inv_dec_id=None):
    inv_bb, inv_enc, inv_dec, detr = None, None, None, None
    if detr_id is not None:
        if detr_id == 'resnet50':
            detr = detr_resnet50(pretrained=True)
        else:
            detr = init_model_from_neptune(run_id=detr_id, module=detr_module)
    if inv_bb_id is not None:
        inv_bb = init_model_from_neptune(run_id=inv_bb_id, module=inv_bb_module)
    if inv_enc_id is not None:
        inv_enc = init_model_from_neptune(run_id=inv_enc_id, module=inv_enc_module)
    if inv_dec_id is not None:
        inv_dec = init_model_from_neptune(run_id=inv_dec_id, module=inv_dec_module)
    return detr, inv_bb, inv_enc, inv_dec


def sequence_to_spatial(seq, h_max_32, w_max_32):
    """
    Transforms a detr transformer encoder input sequence batch (t x b x 256, where t = ceil(h_max/32) * ceil(w_max/32))
    to an inverse backbone input batch (b x 256 x ceil(h_max/32) x ceil(w_max/32))
    """
    spatial = seq.permute(1, 2, 0)
    spatial = spatial.view(spatial.shape[0], spatial.shape[1], h_max_32, w_max_32)
    return spatial
