import torch
import torchvision.transforms as transforms


def bb_emb_to_detr_out(enc_emb, detr, mask, pos):
    memory = bb_emb_to_enc_emb(bb_emb=enc_emb, detr=detr, mask=mask, pos=pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb=memory, detr=detr, mask=mask, pos=pos)
    detr_out = dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
    return detr_out


def dec_emb_to_enc_emb(dec_emb, pos, inv_dec):
    dec_emb = dec_emb[-1]
    dec_pos = inv_dec.query_embed.weight.unsqueeze(1).repeat(1, dec_emb.shape[1], 1)
    tgt = torch.zeros_like(pos)
    hs = inv_dec(tgt, memory=dec_emb, memory_key_padding_mask=None, pos=dec_pos, query_pos=pos)[0]
    return hs


def dec_emb_to_img(dec_emb, mask, pos, inv_dec, inv_enc, inv_bb):
    enc_emb = dec_emb_to_enc_emb(dec_emb, pos, inv_dec)
    bb_emb = enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, mask=mask, pos=pos)
    img = bb_emb_to_img(bb_emb=bb_emb, inv_bb=inv_bb)
    return img


def nested_tensor_to_bb_emb(nested_tensor, detr):
    features, pos = detr.backbone(nested_tensor)
    src, mask = features[-1].decompose()
    bb_emb = detr.input_proj(src).flatten(2).permute(2, 0, 1)
    return bb_emb, pos[-1].flatten(2).permute(2, 0, 1), mask.flatten(1)


def nested_tensor_to_dec_emb(nested_tensor, detr):
    bb_emb, pos, mask = nested_tensor_to_bb_emb(nested_tensor, detr)
    enc_emb = bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    return dec_emb


def dec_emb_to_detr_out(dec_emb, detr):
    dec_emb = dec_emb.transpose(1, 2)
    outputs_class = detr.class_embed(dec_emb)
    outputs_coord = detr.bbox_embed(dec_emb).sigmoid()
    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    return out


def bb_emb_to_enc_emb(bb_emb, detr, mask, pos):
    memory = detr.transformer.encoder(src=bb_emb, src_key_padding_mask=mask, pos=pos)
    return memory


def bb_emb_to_enc_reps(bb_emb, detr, mask, pos):
    int_reps = [bb_emb]
    for layer in detr.transformer.encoder.layers:
        int_reps.append(layer(int_reps[-1], src_key_padding_mask=mask, pos=pos))
    return int_reps


def enc_emb_to_dec_reps(enc_emb, detr, mask, pos):
    query_embed = detr.query_embed.weight
    query_embed = query_embed.unsqueeze(1).repeat(1, enc_emb.shape[1], 1)
    tgt = torch.zeros_like(query_embed)
    hs = detr.transformer.decoder(tgt, enc_emb, memory_key_padding_mask=mask,
                                  pos=pos, query_pos=query_embed)
    int_reps = list(torch.unbind(hs, dim=0))
    int_reps.insert(0, tgt)
    return int_reps


def enc_emb_to_dec_emb(enc_emb, detr, mask, pos):
    query_embed = detr.query_embed.weight
    query_embed = query_embed.unsqueeze(1).repeat(1, enc_emb.shape[1], 1)
    tgt = torch.zeros_like(query_embed)
    hs = detr.transformer.decoder(tgt, enc_emb, memory_key_padding_mask=mask, pos=pos, query_pos=query_embed)
    return hs


def enc_emb_to_bb_emb(enc_emb, inv_enc, mask, pos):
    bb_emb = inv_enc(enc_emb, src_key_padding_mask=mask, pos=pos)
    return bb_emb


def bb_emb_to_img(bb_emb, inv_bb, w_max_32=20, h_max_32=15):
    spatial_rep = sequence_to_spatial(bb_emb, w_max_32=w_max_32, h_max_32=h_max_32)
    img = inv_bb(spatial_rep)
    return img


def enc_emb_to_img(enc_emb, inv_enc, inv_bb, mask, pos):
    bb_emb = enc_emb_to_bb_emb(enc_emb, inv_enc, mask, pos)
    img = bb_emb_to_img(bb_emb, inv_bb)
    return img


def enc_emb_to_detr_out(enc_emb, detr, mask, pos):
    dec_emb = enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
    detr_out = dec_emb_to_detr_out(dec_emb, detr)
    return detr_out


def sequence_to_spatial(seq, h_max_32, w_max_32):
    """
    Transforms a detr transformer encoder input sequence batch (t x b x 256, where t = ceil(h_max/32) * ceil(w_max/32))
    to an inverse backbone input batch (b x 256 x ceil(h_max/32) x ceil(w_max/32))
    """
    spatial = seq.permute(1, 2, 0)
    spatial = spatial.view(spatial.shape[0], spatial.shape[1], h_max_32, w_max_32)
    return spatial


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
        recon_bb_emb = bb_emb_to_img(bb_emb=bb_emb, inv_bb=inv_bb)
    if inv_enc is not None:
        recon_enc_emb = enc_emb_to_img(enc_emb=enc_emb, inv_bb=inv_bb, pos=pos, mask=mask, inv_enc=inv_enc)
    if inv_dec is not None:
        recon_dec_emb = dec_emb_to_img(dec_emb=dec_emb, inv_bb=inv_bb, pos=pos, mask=mask, inv_enc=inv_enc,
                                       inv_dec=inv_dec)
    return recon_bb_emb, recon_enc_emb, recon_dec_emb


@torch.no_grad()
def get_imgs_from_int_reps(int_enc_reps, int_dec_reps, pos, mask, inv_bb, inv_enc, inv_dec):
    int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb = [], [], []
    # reconstructions from intermediate encoder representations used as backbone embedding
    # reconstructions from intermediate encoder representations used as encoder embedding
    # reconstructions from intermediate decoder representations used as decoder embedding
    for rep in int_dec_reps:
        recon = dec_emb_to_img(dec_emb=rep.unsqueeze(0), mask=mask, inv_bb=inv_bb, inv_dec=inv_dec,
                               inv_enc=inv_enc, pos=pos)
        int_recons_from_dec_emb.append(recon)
    for rep in int_enc_reps:
        recon_bb_emb = bb_emb_to_img(bb_emb=rep, inv_bb=inv_bb)
        int_recons_from_bb_emb.append(recon_bb_emb)
        recon_enc_emb = enc_emb_to_img(enc_emb=rep, inv_enc=inv_enc, inv_bb=inv_bb, pos=pos, mask=mask)
        int_recons_from_enc_emb.append(recon_enc_emb)
    return int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb


@torch.no_grad()
def get_int_recons_from_nested_tensor(nested_tensor, detr, inv_bb, inv_enc, inv_dec):
    int_enc_reps, int_dec_reps, pos, mask = get_int_reps_from_nested_tensor(nested_tensor, detr)
    int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb = \
        get_imgs_from_int_reps(int_enc_reps=int_enc_reps, int_dec_reps=int_dec_reps, pos=pos, mask=mask,
                               inv_bb=inv_bb, inv_enc=inv_enc, inv_dec=inv_dec)
    return int_recons_from_bb_emb, int_recons_from_enc_emb, int_recons_from_dec_emb


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def denormalize(img):
    img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).view(-1, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406], device=img.device).view(-1, 1, 1)
    return img
