import config
import torch
from torch.functional import F
import tools.detr_utils as du


def test_inv_enc(inv_enc, detr, dataloader):
    inv_enc.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            x = inputs.to(config.DEVICE)
            with torch.no_grad():
                bb_emb, pos, mask = du.nested_tensor_to_bb_emb(x, detr)
                enc_emb = du.bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
            recon = inv_enc(enc_emb, src_key_padding_mask=mask, pos=pos)
            loss = F.mse_loss(input=recon, target=bb_emb)
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
    return sum_loss / num_inputs
