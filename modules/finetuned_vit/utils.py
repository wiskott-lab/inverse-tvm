import config
import torch
from torch.functional import F
import tools.vit_utils as vu
import tools.coco_utils as cu


def test_finetuned_vit(vit, inv_bb, inv_enc, dataloader, trade_off):
    inv_bb.eval(), inv_enc.eval(), vit.eval()
    sum_chain_loss, sum_class_loss, sum_denormalized_chain_loss, sum_acc = 0, 0, 0, 0
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, target) in enumerate(dataloader):
            tensor, target = tensor.to(config.DEVICE), target.to(config.DEVICE)
            bb_emb = vu.tensor_to_bb_emb(tensor, vit)
            enc_emb = vu.bb_emb_to_enc_emb(bb_emb, vit)

            chain_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
            vit_out = vu.enc_emb_to_vit_out(enc_emb, vit)

            chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=tensor)
            denormalized_chain_recon_loss = F.mse_loss(input=chain_recon, target=cu.denormalize(tensor))


            class_loss = F.cross_entropy(input=vit_out, target=target)
            loss = (1 - trade_off) * class_loss + trade_off * chain_recon_loss
            acc = accuracy(logits=vit_out, target=target)

            sum_acc += acc * len(tensor)
            sum_chain_loss += chain_recon_loss * len(tensor)
            sum_denormalized_chain_loss += denormalized_chain_recon_loss * len(tensor)
            sum_class_loss += class_loss * len(tensor)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)


        return sum_chain_loss / num_inputs

def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e