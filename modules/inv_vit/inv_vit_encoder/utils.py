import config
import torch
from torch.functional import F
import tools.vit_utils as vu

def test_inv_enc(inv_enc, vit, dataloader):
    inv_enc.eval(), vit.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, _) in enumerate(dataloader):
            tensor = tensor.tensors.to(config.DEVICE)
            input_emb = vu.tensor_to_bb_emb(tensor, vit)
            enc_emb = vu.tensor_to_enc_emb(tensor, vit)
            recon = vu.enc_emb_to_bb_emb(enc_emb, inv_enc)
            loss = F.mse_loss(input=recon, target=input_emb)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)
    return sum_loss / num_inputs


def test_inv_sub_enc(inv_sub_enc, vit, dataloader, from_layer, to_layer):
    inv_sub_enc.eval(), vit.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, _) in enumerate(dataloader):
            tensor = tensor.tensors.to(config.DEVICE)
            bb_emb = vu.tensor_to_bb_emb(tensor, vit)
            to_layer_rep = vu.int_enc_rep_to_int_enc_rep(bb_emb, vit, from_layer=0, to_layer=to_layer)
            from_layer_rep = vu.int_enc_rep_to_int_enc_rep(to_layer_rep, vit, from_layer=to_layer, to_layer=from_layer)
            recon = vu.enc_emb_to_bb_emb(from_layer_rep, inv_sub_enc)
            loss = F.mse_loss(input=recon, target=to_layer_rep)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)
    return sum_loss / num_inputs
