import copy

import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.finetuned_detr.utils import test_train_in_parallel, test_train_chain_losses
from tools import training_utils
from tools import detr_utils as du
import tools.neptune_utils as nu
import tools.coco_utils as cu
from modules.inv_detr_bb import models as inv_bb_module
from modules.inv_detr_enc import models as inv_enc_module
from modules.inv_detr_dec import models as inv_dec_module
from modules.inv_detr_pred import models as inv_detector_module

from modules.detr.models import detr as detr_module
from tools.misc_utils import get_parent_file
from timm.optim import create_optimizer_v2
from modules.detr.hubconf import detr_resnet50
from timm.utils.cuda import NativeScaler





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default=None)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=1000)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=32)

    # parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default='OB-333')
    # parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default='OB-320')
    # parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default='OB-325')
    parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default=None)
    parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default=None)
    parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default=None)
    parser.add_argument("--inv_detect_id", "-idet", help='neptune id of inv detector', type=str, default=None)


    args = parser.parse_args()
    epochs = args.epochs

    lr = args.lr
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    inv_bb_id = args.inv_bb_id
    inv_dec_id = args.inv_dec_id
    inv_enc_id = args.inv_enc_id
    inv_detect_id = args.inv_detect_id


    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = training_utils.init_seeds(None)

    model_configs = {'inverse_backbone': {'module_type': 'EnhancedBatchNormalizedConvolutionalDecoder'},
                     'inverse_encoder': {'module_type': 'InverseTransformerEncoder'},
                     'inverse_decoder': {'module_type': 'InverseTransformerDecoder'},
                     'inverse_detector': {'module_type': 'InverseDetector'}}


    optim_configs = {'inverse_backbone': {'module_type': 'adam', 'lr': 0.001},
                     'inverse_encoder': {'module_type': 'adam', 'lr': 0.001},
                     'inverse_decoder': {'module_type': 'adam', 'lr': 0.001},
                     'inverse_detector': {'module_type': 'adam', 'lr': 0.001}}

    # optim_configs = {'inverse_backbone': {'module_type': 'AdamW', 'lr': lr},
    #                  'inverse_encoder': {'module_type': 'AdamW', 'lr': lr},
    #                  'inverse_decoder': {'module_type': 'AdamW', 'lr': lr}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'optim_configs': optim_configs, 'model_configs': model_configs}

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=2*batch_size, dataset_type='test')

    source_files = [str(Path(__file__))]

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

        optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
        models = [inv_bb, inv_enc, inv_dec, detr]
        run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
    else:
        detr = detr_resnet50(pretrained=True)

        if inv_bb_id is not None:
            inv_bb = nu.init_model_from_neptune(run_id=inv_bb_id, module=inv_bb_module)
        else:
            inv_bb = nu.init_model_from_params(module=inv_bb_module, params=neptune_params)

        if inv_enc_id is not None:
            inv_enc = nu.init_model_from_neptune(run_id=inv_enc_id, module=inv_enc_module)
        else:
            inv_enc = nu.init_model_from_params(params=neptune_params, module=inv_enc_module)

        if inv_dec_id is not None:
            inv_dec = nu.init_model_from_neptune(run_id=inv_dec_id, module=inv_dec_module)
        else:
            inv_dec = nu.init_model_from_params(params=neptune_params, module=inv_dec_module)

        if inv_detect_id is not None:
            inv_detect = nu.init_model_from_neptune(run_id=inv_detect_id, module=inv_detector_module)
        else:
            inv_detect = nu.init_model_from_params(params=neptune_params, module=inv_detector_module)


        inv_bb.to(config.DEVICE), inv_enc.to(config.DEVICE), inv_dec.to(config.DEVICE), inv_detect.to(config.DEVICE), detr.to(config.DEVICE)

        best_inv_bb = copy.deepcopy(inv_bb)
        best_inv_enc = copy.deepcopy(inv_enc)
        best_inv_dec = copy.deepcopy(inv_dec)
        best_inv_detect = copy.deepcopy(inv_detect)

        models = [inv_bb, inv_enc, inv_dec, inv_detect]

        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                               capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

        run['params'] = neptune_params
        test_step, train_step, last_epoch = 0, -1, 0
        inv_bb_optim = create_optimizer_v2(model_or_params=inv_bb, lr=optim_configs['inverse_backbone']['lr'], opt=optim_configs['inverse_backbone']['module_type'])
        inv_enc_optim = create_optimizer_v2(model_or_params=inv_enc, lr=optim_configs['inverse_encoder']['lr'], opt=optim_configs['inverse_encoder']['module_type'])
        inv_dec_optim = create_optimizer_v2(model_or_params=inv_dec, lr=optim_configs['inverse_decoder']['lr'], opt=optim_configs['inverse_decoder']['module_type'])
        inv_detect_optim = create_optimizer_v2(model_or_params=inv_detect, lr=optim_configs['inverse_detector']['lr'], opt=optim_configs['inverse_detector']['module_type'])

        optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, inv_detect_optim]

        best_bb, best_enc, best_dec, best_detect = test_train_in_parallel(inv_bb=inv_bb, run=run, detr=detr, dataloader=dataloader_test,
                                       inv_enc=inv_enc, inv_dec=inv_dec, inv_detect=inv_detect)
        test_train_chain_losses(inv_enc=best_inv_enc, inv_dec=best_inv_dec, inv_detect=best_inv_detect,
                                inv_bb=best_inv_bb, run=run, detr=detr, dataloader=dataloader_test)
        nu.upload_model_states(models=models, run=run)
        nu.upload_checkpoint(models=models, optims=optims, best_loss=(best_bb, best_enc, best_dec, best_detect), run=run,
                             test_step=test_step, train_step=train_step)

    bb_scaler = NativeScaler()
    enc_scaler = NativeScaler()
    dec_scaler = NativeScaler()
    detect_scaler = NativeScaler()

    for epoch in range(epochs):
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
                enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, mask=mask, pos=pos, detr=detr)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
                detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
            nested_tensor_recon = du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
            bb_loss = F.mse_loss(input=cu.normalize(nested_tensor_recon), target=nested_tensor.tensors)
            inv_bb_optim.zero_grad()
            bb_scaler(bb_loss, inv_bb_optim, clip_grad=1.0, parameters=inv_bb.parameters())
            run["train/bb_loss"].append(bb_loss.item())

            bb_emb_recon = du.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, pos=pos, mask=mask)
            enc_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
            inv_enc_optim.zero_grad()
            enc_scaler(enc_loss, inv_enc_optim, clip_grad=1.0, parameters=inv_enc.parameters())
            run["train/enc_loss"].append(enc_loss.item())

            enc_emb_recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
            dec_loss = F.mse_loss(input=enc_emb_recon, target=enc_emb)
            inv_dec_optim.zero_grad()
            dec_scaler(dec_loss, inv_dec_optim, clip_grad=1.0, parameters=inv_dec.parameters())
            run["train/dec_loss"].append(dec_loss.item())

            dec_emb_recon = du.detr_out_to_dec_emb(detr_out=detr_out, inv_detect=inv_detect)
            detect_loss = F.mse_loss(input=dec_emb_recon[-1], target=dec_emb[-1])
            inv_detect_optim.zero_grad()
            detect_scaler(detect_loss, inv_detect_optim, clip_grad=1.0, parameters=inv_detect.parameters())
            run["train/detect_loss"].append(detect_loss.item())

            train_step += 1
        bb_val, enc_val, dec_val, detect_val = test_train_in_parallel(inv_bb=inv_bb, run=run, detr=detr,
                                                                      dataloader=dataloader_test, inv_dec=inv_dec,
                                                                      inv_detect=inv_detect, inv_enc=inv_enc)
        test_step += 1
        if bb_val < best_bb:
            nu.upload_model_state(model=inv_bb, run=run)
            best_bb = bb_val
            best_inv_bb = copy.deepcopy(inv_bb)
        if enc_val < best_enc:
            nu.upload_model_state(model=inv_enc, run=run)
            best_enc = enc_val
            best_inv_enc = copy.deepcopy(inv_enc)
        if dec_val < best_dec:
            nu.upload_model_state(model=inv_dec, run=run)
            best_dec = dec_val
            best_inv_dec = copy.deepcopy(inv_dec)

        if detect_val < best_detect:
            nu.upload_model_state(model=inv_detect, run=run)
            best_detect = detect_val
            best_inv_detect = copy.deepcopy(inv_detect)

        test_train_chain_losses(inv_enc=best_inv_enc, inv_dec=best_inv_dec, inv_detect=best_inv_detect,
                                inv_bb=best_inv_bb, run=run, detr=detr, dataloader=dataloader_test)

        nu.upload_checkpoint(models=models, optims=optims, best_loss=(best_bb, best_enc, best_dec, best_detect),
                             run=run, test_step=test_step, train_step=train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()