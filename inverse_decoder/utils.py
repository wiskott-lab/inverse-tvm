import torch
from utils import detr_utils as du
from torch.functional import F


def test_inv_dec(inv_dec, detr, dataloader, device):
    inv_dec.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (img, _) in enumerate(dataloader):
            img = img.to(device)
            with torch.no_grad():
                enc_emb, pos, mask = du.nested_tensor_to_bb_emb(img, detr)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, pos=pos, mask=mask)
            recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
            loss = F.mse_loss(input=recon, target=enc_emb)
            sum_loss += loss * len(img.tensors)
            num_inputs += len(img.tensors)
    return sum_loss / num_inputs
