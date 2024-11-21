import torch
from detr.models.detr import SetCriterion
from detr.models.matcher import HungarianMatcher
import argparse
from torch.functional import F
from detr.hubconf import detr_resnet50
from finetuned_detr.utils import test_finetuned_detr
from inverse_backbone import models as inv_bb_module
from inverse_encoder import models as inv_enc_module
from inverse_decoder import models as inv_dec_module
from utils import detr_utils as du, coco_utils as cu, training_utils
from torch.optim import Adam
from copy import deepcopy

TRAINING_MODE_CHAIN = 'chain'
TRAINING_MODE_SUM = 'sum'
TRAINING_MODE_ISOLATED = 'isolated'
TRAINING_MODE_BACKWARDS_ONLY = 'backwards_only'


def _zero_grad_optims():
    for optim in optims:
        optim.zero_grad()


def _clip_grads():
    if max_norm > 0:
        for model in models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _get_reconstruction_losses():
    enc_emb_recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, pos=pos, inv_dec=inv_dec)
    bb_emb_recon = du.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, mask=mask, pos=pos)
    bb_inputs_recon = du.bb_emb_to_img(bb_emb=bb_emb, inv_bb=inv_bb)
    enc_emb_recon_loss = F.mse_loss(input=enc_emb_recon, target=enc_emb)
    bb_emb_recon_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
    bb_inputs_recon_loss = F.mse_loss(input=du.normalize(bb_inputs_recon), target=nested_tensor.tensors)
    return enc_emb_recon_loss, bb_emb_recon_loss, bb_inputs_recon_loss


def _isolated_step():
    enc_emb_recon_loss, bb_emb_recon_loss, bb_inputs_recon_loss = _get_reconstruction_losses()
    recon_loss = bb_inputs_recon_loss
    loss = (1 - trade_off) * detr_loss + trade_off * recon_loss

    _zero_grad_optims()
    loss.backward(retain_graph=True)
    _clip_grads()
    detr_optim.step()
    inv_bb_optim.step()

    _zero_grad_optims()
    enc_emb_recon_loss.backward(retain_graph=True)
    inv_dec_optim.step()
    _zero_grad_optims()

    bb_emb_recon_loss.backward()
    inv_enc_optim.step()


def _chain_step():
    chain_recon = du.dec_emb_to_img(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc,
                                    inv_bb=inv_bb)
    chain_recon_loss = F.mse_loss(input=du.normalize(chain_recon), target=nested_tensor.tensors)
    loss = (1 - trade_off) * detr_loss + trade_off * chain_recon_loss
    _zero_grad_optims()
    loss.backward(retain_graph=True)
    _clip_grads()
    for optim in optims:
        optim.step()


def _backwards_only_step():
    chain_recon = du.dec_emb_to_img(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc,
                                    inv_bb=inv_bb)
    chain_recon_loss = F.mse_loss(input=du.normalize(chain_recon), target=nested_tensor.tensors)
    loss = (1 - trade_off) * detr_loss + trade_off * chain_recon_loss
    _zero_grad_optims()
    loss.backward(retain_graph=True)
    _clip_grads()
    inv_bb_optim.step()
    inv_enc_optim.step()
    inv_dec_optim.step()


def _sum_step():
    enc_emb_recon_loss, bb_emb_recon_loss, bb_inputs_recon_loss = _get_reconstruction_losses()
    recon_loss = bb_recon_weight * bb_inputs_recon_loss + enc_recon_weight * bb_emb_recon_loss + \
                 dec_recon_weight * enc_emb_recon_loss

    loss = (1 - trade_off) * detr_loss + trade_off * recon_loss
    _zero_grad_optims()
    loss.backward(retain_graph=True)
    _clip_grads()
    inv_bb_optim.step()
    inv_enc_optim.step()
    inv_dec_optim.step()
    detr_optim.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", help='device', type=str, default='cpu')
    parser.add_argument("--path", "-p", help='path for storing trained model', type=str)
    parser.add_argument("--learning_rate_dec", "-lr-dec", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_enc", "-lr-enc", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_bb", "-lr-back", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_detr", "-lr-detr", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=16)
    parser.add_argument("--trade_off", "-t", help='trade-off', type=float, default=0.9)
    parser.add_argument("--bb_recon_weight", "-brw", help='multiplier for bb recon loss', type=float, default=1.0)
    parser.add_argument("--enc_recon_weight", "-erw", help='multiplier for enc recon loss', type=float, default=1.0)
    parser.add_argument("--dec_recon_weight", "-drw", help='multiplier for dec recon loss', type=float, default=1.0)
    parser.add_argument("--eval_loss_id", "-eid", help='eval_loss_id', type=str, default='bb')

    args = parser.parse_args()
    epochs = args.epochs
    path = args.path
    lr_bb = args.learning_rate_bb
    lr_enc = args.learning_rate_enc
    lr_dec = args.learning_rate_dec
    lr_detr = args.learning_rate_detr
    eval_loss_id = args.eval_loss_id
    trade_off = args.trade_off
    batch_size = args.batch_size
    device = args.device
    detr_id = args.detr_id
    training_mode = args.training_mode
    bb_recon_weight = args.bb_recon_weight
    enc_recon_weight = args.enc_recon_weight
    dec_recon_weight = args.dec_recon_weight
    cost_class, cost_bbox, cost_giou = 1.0, 5.0, 2.0  # default values from detr main
    eos_coef = 0.1  # default values from detr main
    max_norm = 0.1
    weight_dict = {'loss_ce': cost_class, 'loss_bbox': cost_bbox, 'loss_giou': cost_giou}
    losses = ['labels', 'boxes', 'cardinality']

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = training_utils.init_seeds(None)

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')

    matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
    criterion = SetCriterion(num_classes=91, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)

    criterion.to(device)
    detr = detr_resnet50(pretrained=True)
    inv_bb = inv_bb_module.EnhancedBatchNormalizedConvolutionalDecoder()
    inv_enc = inv_enc_module.InverseTransformerEncoder()
    inv_dec = inv_dec_module.InverseTransformerDecoder()
    inv_bb.to(device), inv_enc.to(device), inv_dec.to(device), detr.to(device)
    models = [inv_bb, inv_enc, inv_dec, detr]
    torch.save(deepcopy(models), path)

    inv_bb_optim = Adam(params=inv_bb.parameters(), lr=lr_bb)
    inv_enc_optim = Adam(params=inv_enc.parameters(), lr=lr_bb)
    inv_dec_optim = Adam(params=inv_dec.parameters(), lr=lr_bb)
    detr_optim = Adam(params=detr.parameters(), lr=lr_bb)

    optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
    best_loss = test_finetuned_detr(inv_bb=inv_bb, detr=detr, dataloader=dataloader_test, inv_enc=inv_enc,
                                    criterion=criterion, inv_dec=inv_dec, eval_loss_id=eval_loss_id, device=device)

    for epoch in range(epochs):
        for batch_id, (nested_tensor, target) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
            enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, detr=detr, mask=mask, pos=pos)
            dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
            detr_outputs = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)

            loss_dict = criterion(detr_outputs, target)
            weight_dict = criterion.weight_dict
            detr_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            if training_mode == TRAINING_MODE_SUM:
                _sum_step()
            elif training_mode == TRAINING_MODE_CHAIN:
                _chain_step()
            elif training_mode == TRAINING_MODE_ISOLATED:
                _isolated_step()
            elif training_mode == TRAINING_MODE_BACKWARDS_ONLY:
                _backwards_only_step()
            else:
                NameError(f'Unknown training mode: {training_mode}')

        eval_loss = test_finetuned_detr(inv_bb=inv_bb, detr=detr, dataloader=dataloader_test, inv_enc=inv_enc,
                                        criterion=criterion, inv_dec=inv_dec, eval_loss_id=eval_loss_id, device=device)
        if eval_loss < best_loss:
            torch.save(deepcopy(models), path)
            best_loss = eval_loss
