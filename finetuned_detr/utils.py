import torch
from torch.functional import F
from utils import detr_utils as du


@torch.no_grad()
def test_finetuned_detr(inv_bb, inv_enc, inv_dec, detr, criterion, dataloader, device, eval_loss_id='bb'):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), detr.eval(), criterion.eval()

    sum_class_loss, sum_detr_loss, sum_bb_recon_loss, sum_chain_recon_loss, num_inputs = 0, 0, 0, 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(device)
        bb_emb, pos, mask = du.nested_tensor_to_bb_emb(img, detr)
        enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, detr=detr, mask=mask, pos=pos)
        dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
        chain_recon = du.dec_emb_to_img(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos,  inv_enc=inv_enc,
                                        inv_bb=inv_bb)
        chain_recon_loss = F.mse_loss(input=du.normalize(chain_recon), target=img.tensors)
        sum_chain_recon_loss += chain_recon_loss * len(inputs.tensors)
        num_inputs += len(inputs.tensors)

    if eval_loss_id == 'chain':
        return sum_chain_recon_loss / num_inputs
    elif eval_loss_id == 'bb':
        return sum_bb_recon_loss / num_inputs
    else:
        raise NameError(f"Unknown eval_loss_id {eval_loss_id}")
