import torch
import config
from modules.detr.models.detr import SetCriterion
from modules.detr.models.matcher import HungarianMatcher
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.finetuned_detr.utils import test_finetuned_detr
from tools import training_utils
from tools import detr_utils as du
import tools.neptune_utils as nu
import tools.coco_utils as cu
from modules.inv_detr_bb import models as inv_bb_module
from modules.inv_detr_enc import models as inv_enc_module
from modules.inv_detr_dec import models as inv_dec_module
from modules.detr.models import detr as detr_module
from tools.misc_utils import get_parent_file

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

    run["train/loss"].append(loss)
    run["train/recon_loss"].append(recon_loss)
    run["train/bb_recon_loss"].append(bb_inputs_recon_loss)
    run["train/enc_recon_loss"].append(bb_emb_recon_loss)
    run["train/dec_recon_loss"].append(enc_emb_recon_loss)


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
    run["train/loss"].append(loss)
    run["train/recon_loss"].append(chain_recon_loss)


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
    run["train/loss"].append(loss)
    run["train/recon_loss"].append(chain_recon_loss)


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

    run["train/loss"].append(loss)
    run["train/recon_loss"].append(recon_loss)
    run["train/bb_recon_loss"].append(bb_inputs_recon_loss)
    run["train/enc_recon_loss"].append(bb_emb_recon_loss)
    run["train/dec_recon_loss"].append(enc_emb_recon_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default=None)

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
    parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default='OB-333')
    parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default='OB-320')
    parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default='OB-325')
    parser.add_argument("--detr_run_id", "-did", help='neptune id of inv bb', type=str, default='resnet50')
    parser.add_argument("--eval_loss_id", "-eid", help='eval_loss_id', type=str, default='bb')

    args = parser.parse_args()
    epochs = args.epochs

    lr_bb = args.learning_rate_bb
    lr_enc = args.learning_rate_enc
    lr_dec = args.learning_rate_dec
    lr_detr = args.learning_rate_detr

    eval_loss_id = args.eval_loss_id
    trade_off = args.trade_off
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device
    inv_bb_id = args.inv_bb_id
    inv_dec_id = args.inv_dec_id
    inv_enc_id = args.inv_enc_id
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

    model_configs = {'inv_bb': {'as_in': inv_bb_id},
                     'inv_enc': {'as_in': inv_enc_id},
                     'inv_dec': {'as_in': inv_dec_id},
                     'detr': {'as_in': detr_id}}

    optim_configs = {'inv_bb': {'module_type': 'AdamW', 'lr': lr_bb},
                     'inv_enc': {'module_type': 'AdamW', 'lr': lr_enc},
                     'inv_dec': {'module_type': 'AdamW', 'lr': lr_dec},
                     'detr': {'module_type': 'AdamW', 'lr': lr_detr}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'optim_configs': optim_configs, 'model_configs': model_configs,
                      'init_detr': detr_id, 'init_inv_bb': inv_bb_id,
                      'init_inv_enc': inv_enc_id, 'init_inv_dec': inv_dec_id,
                      'trade_off': trade_off, 'cost_class': cost_class,
                      'cost_bbox': cost_bbox, 'cost_giou': cost_giou, 'eos_coef': eos_coef, 'max_norm': max_norm,
                      'bb_recon_weight': bb_recon_weight, 'enc_recon_weight': enc_recon_weight,
                      'dec_recon_weight': dec_recon_weight, 'training_mode': training_mode,
                      'eval_loss_id': eval_loss_id}

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='val')

    source_files = [str(Path(__file__))]

    matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
    criterion = SetCriterion(num_classes=91, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=eos_coef, losses=losses)
    criterion.to(config.DEVICE)

    if run_id:
        neptune_params = nu.get_params(run_id=run_id)
        checkpoint = nu.get_checkpoint(run_id=run_id)
        inv_bb, inv_bb_optim = nu.init_from_checkpoint(checkpoint, inv_bb_module, neptune_params)
        detr, detr_optim = nu.init_from_checkpoint(checkpoint, detr_module, neptune_params)
        inv_enc, inv_enc_optim = nu.init_from_checkpoint(checkpoint, inv_enc_module, neptune_params)
        inv_dec, inv_dec_optim = nu.init_from_checkpoint(checkpoint, inv_dec_module, neptune_params)
        best_loss = checkpoint['best_loss']
        test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        last_epoch = neptune_params['epochs']
        cost_class, cost_bbox, cost_giou = neptune_params['cost_class'], neptune_params['cost_bbox'], \
            neptune_params['cost_giou']
        eos_coef, max_norm = neptune_params['eos_coef'], neptune_params['max_norm']
        trade_off = neptune_params['trade_off']
        bb_recon_weight = neptune_params['bb_recon_weight']
        enc_recon_weight = neptune_params['enc_recon_weight']
        dec_recon_weight = neptune_params['dec_recon_weight']
        training_mode = neptune_params['training_mode']

        train_detr_in_eval_mode = neptune_params['train_detr_in_eval_mode']
        train_inv_bb_in_eval_mode = neptune_params['train_inv_bb_in_eval_mode']
        train_inv_enc_in_eval_mode = neptune_params['train_inv_enc_in_eval_mode']
        train_inv_dec_in_eval_mode = neptune_params['train_inv_dec_in_eval_mode']
        train_set_criterion_in_eval_mode = neptune_params['train_set_criterion_in_eval_mode']

        weight_dict = {'loss_ce': cost_class, 'loss_bbox': cost_bbox, 'loss_giou': cost_giou}
        losses = ['labels', 'boxes', 'cardinality']

        matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        criterion = SetCriterion(num_classes=91, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=eos_coef, losses=losses)
        criterion.to(config.DEVICE)

        optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
        models = [inv_bb, inv_enc, inv_dec, detr]
        run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
    else:
        detr, inv_bb, inv_enc, inv_dec = du.init_detr_modules(inv_bb_id=inv_bb_id, inv_enc_id=inv_enc_id,
                                                              inv_dec_id=inv_dec_id, detr_id=detr_id)

        inv_bb.to(config.DEVICE), inv_enc.to(config.DEVICE), inv_dec.to(config.DEVICE), detr.to(config.DEVICE)

        inv_bb_optim = nu.init_optim_from_params(inv_bb, neptune_params)
        inv_enc_optim = nu.init_optim_from_params(inv_enc, neptune_params)
        inv_dec_optim = nu.init_optim_from_params(inv_dec, neptune_params)
        detr_optim = nu.init_optim_from_params(detr, neptune_params)

        nu.update_model_configs(params=neptune_params, modules=[inv_bb_module, inv_enc_module, inv_dec_module],
                                run_ids=[inv_bb_id, inv_enc_id, inv_dec_id])

        optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
        models = [inv_bb, inv_enc, inv_dec, detr]
        run = neptune.init_run(project=config.PROJECT, source_files=source_files, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        run['params'] = neptune_params
        best_loss = test_finetuned_detr(inv_bb=inv_bb, run=run, detr=detr, dataloader=dataloader_test, inv_enc=inv_enc,
                                        criterion=criterion, inv_dec=inv_dec, eval_loss_id=eval_loss_id)
        test_step, train_step, last_epoch = 0, -1, 0

        nu.upload_model_states(models=models, run=run)
        nu.upload_checkpoint(models=models, optims=optims, best_loss=best_loss, run=run,
                             test_step=test_step, train_step=train_step)

    for epoch in range(epochs):
        for batch_id, (nested_tensor, target) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            target = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in target]

            bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
            enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, detr=detr, mask=mask, pos=pos)
            dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
            detr_outputs = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)

            loss_dict = criterion(detr_outputs, target)
            weight_dict = criterion.weight_dict
            detr_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            run["train/class_loss"].append(loss_dict['loss_ce'])
            run["train/bbox_loss"].append(loss_dict['loss_bbox'])
            run["train/giou_loss"].append(loss_dict['loss_giou'])
            run["train/detr_loss"].append(detr_loss)
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

            train_step += 1

        eval_loss = test_finetuned_detr(inv_bb=inv_bb, run=run, detr=detr, dataloader=dataloader_test, inv_enc=inv_enc,
                                        criterion=criterion, inv_dec=inv_dec, eval_loss_id=eval_loss_id)
        test_step += 1
        if eval_loss < best_loss:
            nu.upload_model_states(models=models, run=run)
            best_loss = eval_loss
        nu.upload_checkpoint(models=models, optims=optims, best_loss=best_loss, run=run,
                             test_step=test_step, train_step=train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()