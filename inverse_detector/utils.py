import torch
from torch.functional import F
from utils import detr_utils as du


def test_inverse_detector(model, detr, dataloader, device):
    model.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            x = inputs.to(device)
            bb_emb, pos, mask = du.nested_tensor_to_bb_emb(x, detr)
            enc_emb = du.bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
            dec_emb = du.enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
            detr_out = du.dec_emb_to_detr_out(dec_emb, detr)
            recon = model(detr_out["pred_logits"], detr_out["pred_boxes"])
            loss = F.mse_loss(input=recon, target=dec_emb[-1].transpose(0, 1))
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
    return sum_loss / num_inputs
