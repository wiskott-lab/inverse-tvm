import config
import torch
from torch.functional import F
import tools.vit_utils as vu
import tools.coco_utils as cu

def test_inv_bb(inv_bb, vit, dataloader):
    inv_bb.eval(), vit.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, _) in enumerate(dataloader):
            tensor = tensor.tensors.to(config.DEVICE)
            input_emb = vu.tensor_to_bb_emb(tensor, vit)
            recon = cu.normalize(vu.bb_emb_to_tensor(input_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=tensor)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)
    return sum_loss / num_inputs
