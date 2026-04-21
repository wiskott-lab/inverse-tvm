import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.finetuned_detr.utils import test_classic_in_par
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


def init_submodules(params, run_ids, modules):
    model = []
    for run_id, module in zip(run_ids, modules):
        if run_id is None:
            sub_model = nu.init_model_from_params(module=module, params=params)
        else:
            sub_model = nu.init_model_from_neptune(module=module, run_id=run_id)
        model.append(sub_model)
    return tuple(model)


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
                     'inverse_decoder': {'module_type': 'adam', 'lr': 0.000001},
                     'inverse_detector': {'module_type': 'adam', 'lr': 0.000001}}



    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')

    source_files = [str(Path(__file__))]

    train_bb_to_img, train_enc_to_img, train_dec_to_img, train_detr_out_to_img = True, True, True, True
    bb_to_img_optim, enc_to_img_optim, dec_to_img_optim, detr_out_to_img_optim = None, None, None, None
    bb_to_img, enc_to_img, dec_to_img, detr_out_to_img = None, None, None, None

    init_bb_id, init_enc_id, init_dec_id, init_detect_id = None, None, None, None  # bb = 'OB-333'

    bb_ids = (init_bb_id, )
    bb_modules = (inv_bb_module, )

    enc_ids = (init_bb_id, init_enc_id)
    enc_modules = (inv_bb_module, inv_enc_module)

    dec_ids = (init_bb_id, init_enc_id, init_dec_id)
    dec_modules = (inv_bb_module, inv_enc_module, inv_dec_module)

    detect_ids = (init_bb_id, init_enc_id, init_dec_id, init_detect_id)
    detect_modules = (inv_bb_module, inv_enc_module, inv_dec_module, inv_detector_module)

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'optim_configs': optim_configs, 'model_configs': model_configs,
                      'init_bb_id': init_bb_id, 'init_enc_id': init_enc_id, 'init_dec_id': init_dec_id,
                      'init_detect_id': init_detect_id}

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
        detr.to(config.DEVICE)
        bb_to_img = init_submodules(params=neptune_params, run_ids=bb_ids, modules=bb_modules)
        for module in bb_to_img:
            module.to(config.DEVICE)
        bb_to_img_params = bb_to_img[0].parameters()
        bb_to_img_optim = create_optimizer_v2(model_or_params=bb_to_img_params,
                                              lr=optim_configs['inverse_backbone']['lr'],
                                              opt=optim_configs['inverse_backbone']['module_type'])
        enc_to_img = init_submodules(params=neptune_params, run_ids=enc_ids, modules=enc_modules)
        for module in enc_to_img:
            module.to(config.DEVICE)
        enc_to_img_params = list(enc_to_img[0].parameters()) + list(enc_to_img[1].parameters())
        enc_to_img_optim = create_optimizer_v2(model_or_params=enc_to_img_params,
                                               lr=optim_configs['inverse_encoder']['lr'],
                                               opt=optim_configs['inverse_encoder']['module_type'])

        dec_to_img = init_submodules(params=neptune_params, run_ids=dec_ids, modules=dec_modules)
        for module in dec_to_img:
            module.to(config.DEVICE)
        dec_to_img_params =list(dec_to_img[0].parameters()) + list(dec_to_img[1].parameters()) + list(dec_to_img[2].parameters())
        dec_to_img_optim = create_optimizer_v2(model_or_params=dec_to_img_params,
                                               lr=optim_configs['inverse_decoder']['lr'],
                                               opt=optim_configs['inverse_decoder']['module_type'])

        detr_out_to_img = init_submodules(params=neptune_params, modules=detect_modules, run_ids=detect_ids)
        for module in detr_out_to_img:
            module.to(config.DEVICE)
        detr_out_to_img_params = list(detr_out_to_img[0].parameters()) + list(detr_out_to_img[1].parameters()) + \
                                 list(detr_out_to_img[2].parameters()) + list(detr_out_to_img[3].parameters())

        detr_out_to_img_optim = create_optimizer_v2(model_or_params=detr_out_to_img_params,
                                                    lr=optim_configs['inverse_detector']['lr'],
                                                    opt=optim_configs['inverse_detector']['module_type'])

        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                               capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

        run['params'] = neptune_params
        test_step, train_step, last_epoch = 0, -1, 0

        optims = [bb_to_img_optim, enc_to_img_optim, dec_to_img_optim, detr_out_to_img_optim]
        models =  [bb_to_img, enc_to_img, dec_to_img, detr_out_to_img]
        #
        best_bb, best_enc, best_dec, best_detect = test_classic_in_par(run=run, detr=detr, bb_to_img=bb_to_img,
                                                                       enc_to_img=enc_to_img, dec_to_img=dec_to_img,
                                                                       detr_out_to_img=detr_out_to_img,
                                                                       dataloader=dataloader_test)
        nu.upload_model_state_tuple(models=bb_to_img, run=run, model_id='bb')
        nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')
        nu.upload_model_state_tuple(models=dec_to_img, run=run, model_id='dec')
        nu.upload_model_state_tuple(models=detr_out_to_img, run=run, model_id='detect')

        nu.upload_checkpoint_classic(models=models, optims=optims, best_loss=(best_bb, best_enc, best_dec, best_detect),
                             run=run, test_step=test_step, train_step=train_step)

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
            if train_bb_to_img:
                bb_to_img_recon = du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=bb_to_img[0])
                bb_loss = F.mse_loss(input=cu.normalize(bb_to_img_recon), target=nested_tensor.tensors)
                bb_to_img_optim.zero_grad()
                bb_scaler(bb_loss, bb_to_img_optim, clip_grad=1.0, parameters=bb_to_img_params)
                run["train/bb_loss"].append(bb_loss.item())
            if train_enc_to_img:
                enc_to_img_recon = du.enc_emb_to_img_tensor(enc_emb=enc_emb, inv_bb=enc_to_img[0],
                                                            inv_enc=enc_to_img[1], mask=mask, pos=pos)
                enc_loss = F.mse_loss(input=cu.normalize(enc_to_img_recon), target=nested_tensor.tensors)
                enc_to_img_optim.zero_grad()
                enc_scaler(enc_loss, enc_to_img_optim, clip_grad=1.0, parameters=enc_to_img_params)
                run["train/enc_loss"].append(enc_loss.item())
            if train_dec_to_img:
                dec_to_img_recon = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_bb=dec_to_img[0],
                                                            inv_enc=dec_to_img[1], inv_dec=dec_to_img[2], mask=mask,
                                                            pos=pos)
                dec_loss = F.mse_loss(input=cu.normalize(dec_to_img_recon), target=nested_tensor.tensors)
                dec_to_img_optim.zero_grad()
                dec_scaler(dec_loss, dec_to_img_optim, clip_grad=1.0, parameters=dec_to_img_params)
                run["train/dec_loss"].append(dec_loss.item())
            if train_detr_out_to_img:
                detr_out_to_img_recon = du.detr_out_to_img_tensor(detr_out=detr_out, inv_bb=detr_out_to_img[0],
                                                                  inv_enc=detr_out_to_img[1],
                                                                  inv_dec=detr_out_to_img[2],
                                                                  inv_detect=detr_out_to_img[3], mask=mask, pos=pos)
                detect_loss = F.mse_loss(input=cu.normalize(detr_out_to_img_recon), target=nested_tensor.tensors)
                detr_out_to_img_optim.zero_grad()
                detect_scaler(detect_loss, detr_out_to_img_optim, clip_grad=1.0, parameters=detr_out_to_img_params)
                run["train/detect_loss"].append(detect_loss.item())
            train_step += 1
        bb_val, enc_val, dec_val, detect_val = test_classic_in_par(run=run, detr=detr, bb_to_img=bb_to_img,
                                                                   enc_to_img=enc_to_img, dec_to_img=dec_to_img,
                                                                   detr_out_to_img=detr_out_to_img,
                                                                   dataloader=dataloader_test)
        test_step += 1
        if train_bb_to_img and bb_val < best_bb:
            nu.upload_model_state_tuple(models=bb_to_img, run=run,  model_id='bb')
            best_bb = bb_val
        if train_enc_to_img and enc_val < best_enc:
            nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')
            best_enc = enc_val

        if train_dec_to_img and dec_val < best_dec:
            nu.upload_model_state_tuple(models=dec_to_img, run=run,  model_id='dec')
            best_dec = dec_val

        if train_detr_out_to_img and detect_val < best_detect:
            nu.upload_model_state_tuple(models=detr_out_to_img, run=run, model_id='detect')
            best_detect = detect_val

        nu.upload_checkpoint_classic(models=models, optims=optims, best_loss=(best_bb, best_enc, best_dec, best_detect),
                                     run=run, test_step=test_step, train_step=train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()

