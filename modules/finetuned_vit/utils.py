import config
import torch
from torch.functional import F
import tools.vit_utils as vu
import tools.coco_utils as cu
import wandb
from tools.coco_utils import denormalize


def test_finetuned_vit(vit, inv_bb, inv_enc, dataloader, trade_off, step=None):
    inv_bb.eval(), inv_enc.eval(), vit.eval()
    sum_chain_loss, sum_class_loss, sum_acc = 0, 0, 0
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, target) in enumerate(dataloader):
            tensor, target = tensor.to(config.DEVICE), target.to(config.DEVICE)
            bb_emb = vu.tensor_to_bb_emb(tensor, vit)
            enc_emb = vu.bb_emb_to_enc_emb(bb_emb, vit)
            chain_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
            vit_out = vu.enc_emb_to_vit_out(enc_emb, vit)
            chain_recon_loss = F.mse_loss(input=chain_recon, target=tensor)

            class_loss = F.cross_entropy(input=vit_out, target=target)
            loss = (1 - trade_off) * class_loss + trade_off * chain_recon_loss
            acc = accuracy(logits=vit_out, target=target)

            sum_acc += acc * len(tensor)
            sum_chain_loss += chain_recon_loss * len(tensor)
            sum_class_loss += class_loss * len(tensor)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)

        wandb.log({f'val/loss': (sum_loss  / num_inputs).item()}, step=step)
        wandb.log({f'val/class_loss': (sum_class_loss  / num_inputs).item()}, step=step)
        wandb.log({f'val/recon_loss': (sum_chain_loss  / num_inputs).item()}, step=step)
        wandb.log({f'val/accuracy': (sum_acc  / num_inputs).item()}, step=step)

        return (sum_loss / num_inputs).item()


def test_finetuned_vit_for_recons(vit, inv_bb, inv_enc, dataloader, run=None):
    inv_bb.eval(), inv_enc.eval(), vit.eval()
    sum_chain_loss, sum_denormalized_chain_loss = 0, 0
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (tensor, target) in enumerate(dataloader):
            tensor, target = tensor.to(config.DEVICE), target.to(config.DEVICE)
            bb_emb = vu.tensor_to_bb_emb(tensor, vit)
            enc_emb = vu.bb_emb_to_enc_emb(bb_emb, vit)

            chain_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
            chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=tensor)

            sum_chain_loss += chain_recon_loss * len(tensor)
            num_inputs += len(tensor)
        if run:
            run["val/recon_loss"].append(sum_chain_loss / num_inputs)
        return sum_chain_loss / num_inputs


def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


@torch.no_grad()
def test_train_chain_losses(vit, inv_bb, inv_enc, dataloader, run=None, normalize=False):
    inv_bb.eval(), inv_enc.eval(), vit.eval()
    num_inputs = 0
    sum_chain_bb_loss, sum_chain_enc_loss = 0, 0

    for batch_id, (img, _) in enumerate(dataloader):
        img = img.to(config.DEVICE)
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)
        # vit_out = vu.enc_emb_to_vit_out(enc_emb=enc_emb, vit=vit)

        chain_bb_emb = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
        chain_enc_emb = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)

        if normalize:
            img = cu.denormalize(img)

        bb_loss = F.mse_loss(input=chain_bb_emb, target=img)
        enc_loss = F.mse_loss(input=chain_enc_emb, target=img)
        sum_chain_bb_loss += bb_loss * len(img)
        sum_chain_enc_loss += enc_loss * len(img)

        num_inputs += len(img)

    if run:
        run["test/bb_loss_chain"].append((sum_chain_bb_loss / num_inputs).item())
        run["test/enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())



@torch.no_grad()
def test_train_chain_losses_sub(vit, inv_bb, inv_sub_enc_1, inv_sub_enc_2, dataloader, run=None, normalize=False):
    inv_bb.eval(), inv_sub_enc_1.eval(), inv_sub_enc_2.eval(), vit.eval()
    num_inputs = 0
    sum_chain_bb_loss, sum_chain_enc_sub_loss, sum_chain_enc_loss = 0, 0, 0

    for batch_id, (img, _) in enumerate(dataloader):
        img = img.to(config.DEVICE)
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        sub_enc_emb = vu.bb_emb_to_sub_enc_emb(bb_emb=bb_emb, vit=vit)
        enc_emb = vu.sub_enc_emb_to_enc_emb(sub_enc_emb=sub_enc_emb, vit=vit)        # vit_out = vu.enc_emb_to_vit_out(enc_emb=enc_emb, vit=vit)

        chain_bb_emb = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=inv_bb)

        chain_sub_enc_emb = vu.sub_enc_emb_to_bb_emb(sub_enc_emb=sub_enc_emb, inv_sub_enc=inv_sub_enc_1)
        chain_sub_enc_emb = vu.bb_emb_to_tensor(bb_emb=chain_sub_enc_emb, inv_bb=inv_bb)

        chain_enc_emb = vu.sub_enc_emb_to_bb_emb(sub_enc_emb=enc_emb, inv_sub_enc=inv_sub_enc_2)
        chain_enc_emb = vu.sub_enc_emb_to_bb_emb(sub_enc_emb=chain_enc_emb, inv_sub_enc=inv_sub_enc_1)
        chain_enc_emb = vu.bb_emb_to_tensor(bb_emb=chain_enc_emb, inv_bb=inv_bb)

        if normalize:
            img = cu.denormalize(img)

        bb_loss = F.mse_loss(input=chain_bb_emb, target=img)
        sub_enc_loss = F.mse_loss(input=chain_sub_enc_emb, target=img)
        enc_loss = F.mse_loss(input=chain_enc_emb, target=img)

        sum_chain_bb_loss += bb_loss * len(img)
        sum_chain_enc_loss += enc_loss * len(img)
        sum_chain_enc_sub_loss += sub_enc_loss * len(img)
        num_inputs += len(img)

    if run:
        run["test/bb_loss_chain"].append((sum_chain_bb_loss / num_inputs).item())
        run["test/sub_enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())
        run["test/enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())


@torch.no_grad()
def test_train_in_parallel_sub(vit, inv_bb, inv_sub_enc_1, inv_sub_enc_2, dataloader, run=None, normalize=False):
    inv_bb.eval(), inv_sub_enc_1.eval(), inv_sub_enc_2.eval(), vit.eval()
    sum_bb_loss, sum_sub_enc_1_loss, sum_sub_enc_2_loss, num_inputs = 0, 0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        sub_enc_emb = vu.bb_emb_to_sub_enc_emb(bb_emb=bb_emb, vit=vit)
        enc_emb = vu.sub_enc_emb_to_enc_emb(sub_enc_emb=sub_enc_emb, vit=vit)

        img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
        if normalize:
            img_recon = cu.normalize(img_recon)
        bb_emb_recon = vu.sub_enc_emb_to_bb_emb(sub_enc_emb=sub_enc_emb, inv_sub_enc=inv_sub_enc_1)
        sub_enc_emb_recon = vu.enc_emb_to_sub_enc_emb(enc_emb=enc_emb, inv_sub_enc=inv_sub_enc_2)

        bb_loss = F.mse_loss(input=img_recon, target=img)
        sub_enc_1_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
        sub_enc_2_loss = F.mse_loss(input=sub_enc_emb_recon, target=sub_enc_emb)


        num_inputs += len(img)
        sum_bb_loss += bb_loss * len(img)
        sum_sub_enc_1_loss += sub_enc_1_loss * len(img)
        sum_sub_enc_2_loss += sub_enc_2_loss * len(img)


    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/sub_enc_1_loss"].append((sum_sub_enc_1_loss / num_inputs).item())
        run["test/sub_enc_2_loss"].append((sum_sub_enc_2_loss / num_inputs).item())

    return (sum_bb_loss / num_inputs).item(), (sum_sub_enc_1_loss / num_inputs).item(), (sum_sub_enc_2_loss / num_inputs).item()

@torch.no_grad()
def test_train_in_parallel(vit, inv_bb, inv_enc, dataloader, run=None, normalize=False):
    inv_bb.eval(), inv_enc.eval(), vit.eval()
    sum_bb_loss, sum_enc_loss, num_inputs = 0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)

        img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
        if normalize:
            img_recon = cu.normalize(img_recon)
        bb_emb_recon = vu.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc)
        bb_loss = F.mse_loss(input=img_recon, target=img)
        enc_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)

        num_inputs += len(img)
        sum_bb_loss += bb_loss * len(img)
        sum_enc_loss += enc_loss * len(img)

    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/enc_loss"].append((sum_enc_loss / num_inputs).item())


        # run["test/bb_loss_chain"].append((sum_chain_bb_loss / num_inputs).item())
        # run["test/enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())
        # run["test/dec_loss_chain"].append((sum_chain_dec_loss / num_inputs).item())
        # run["test/detect_loss_chain"].append((sum_chain_detect_loss / num_inputs).item())
        #
        # run["test/bb_loss_chain_denorm"].append((sum_chain_bb_loss_denorm / num_inputs).item())
        # run["test/enc_loss_chain_denorm"].append((sum_chain_enc_loss_denorm / num_inputs).item())
        # run["test/dec_loss_chain_denorm"].append((sum_chain_dec_loss_denorm / num_inputs).item())
        # run["test/detect_loss_chain_denorm"].append((sum_chain_detect_loss_denorm / num_inputs).item())
    return (sum_bb_loss / num_inputs).item(), (sum_enc_loss / num_inputs).item()


@torch.no_grad()
def test_classic_in_par(vit, bb_to_img, enc_to_img, dataloader, run=None, normalize=False):
    sum_bb_loss, sum_enc_loss,  num_inputs = 0, 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        if normalize:
            denormalized_img = cu.denormalize(img)
        else:
            denormalized_img = img
        bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
        enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)

        if bb_to_img is not None:
            bb_to_img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=bb_to_img[0])
            bb_loss = F.mse_loss(input=bb_to_img_recon, target=denormalized_img)
            sum_bb_loss += bb_loss * len(img)
        if enc_to_img is not None:
            enc_to_img_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_bb=enc_to_img[0], inv_enc=enc_to_img[1])
            enc_loss = F.mse_loss(input=enc_to_img_recon, target=denormalized_img)
            sum_enc_loss += enc_loss * len(img)
        num_inputs += len(img)
    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/enc_loss"].append((sum_enc_loss / num_inputs).item())
    return (sum_bb_loss / num_inputs).item(), (sum_enc_loss / num_inputs).item()
