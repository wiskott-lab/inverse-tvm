import torch
from torch.functional import F
from utils import detr_utils as du


def test_inv_bb(inv_bb, detr, dataloader, device):
    inv_bb.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            samples = inputs.to(device)
            bb_emb, _, _ = du.nested_tensor_to_bb_emb(samples, detr)
            recon = du.normalize(du.bb_emb_to_img(bb_emb=bb_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=samples.tensors)
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
    return sum_loss / num_inputs
