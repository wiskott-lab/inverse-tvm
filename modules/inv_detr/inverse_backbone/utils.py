import config
import torch
from torch.functional import F
from tools import detr_utils
import tools.coco_utils as cu


def test_inv_bb(inv_bb, detr, dataloader):
    inv_bb.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            samples = inputs.to(config.DEVICE)
            bb_emb, _, _ = detr_utils.nested_tensor_to_bb_emb(samples, detr)
            recon = cu.normalize(detr_utils.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=samples.tensors)
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
    return sum_loss / num_inputs
