import copy
from torchvision import transforms, datasets

import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune

from modules.finetuned_vit.utils import test_train_in_parallel, test_train_chain_losses
from tools import training_utils
import tools.neptune_utils as nu
import tools.coco_utils as cu
from modules.inv_vit_bb import models as inv_bb_module
from modules.inv_vit_enc import models as inv_enc_module
from torch.utils.data import DataLoader
from modules.detr.models import detr as detr_module
from tools.misc_utils import get_parent_file
from timm.optim import create_optimizer_v2
from timm.utils.cuda import NativeScaler
import tools.vit_utils as vu

import timm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default=None)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=1000)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=64)

    # parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default='OB-333')
    # parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default='OB-320')
    # parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default='OB-325')
    parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default=None)
    parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default=None)

    args = parser.parse_args()
    epochs = args.epochs

    lr = args.lr
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    inv_bb_id = args.inv_bb_id
    inv_enc_id = args.inv_enc_id

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = training_utils.init_seeds(None)

    model_configs = {'inv_vit_bb': {'module_type': 'EnhancedVitBackbone'},
                     'inv_vit_encoder': {'module_type': 'InverseViTEncoder'}}

    optim_configs = {'inv_vit_bb': {'module_type': 'adam', 'lr': 0.001},
                     'inv_vit_encoder': {'module_type': 'adam', 'lr': 0.0001}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'imagenet1k', 'model_configs': model_configs, 'optim_configs': optim_configs,
                      'init_vit': 'deit3_base_patch16_224'}  # deit3_base_patch16_224, vit_base_patch16_224


    if neptune_params['init_vit'] == 'deit3_base_patch16_224':
        transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        normalize = True
    else:
        transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
        transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
        normalize = False

    dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
                                  num_workers=4, shuffle=True)

    dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=int(8 * args.batch_size), drop_last=False, pin_memory=True,
                                num_workers=4, shuffle=False)

    source_files = [str(Path(__file__))]

    if run_id:
        neptune_params = nu.get_params(run_id=run_id)
        # checkpoint = nu.get_checkpoint(run_id=run_id)
        # inv_bb, inv_bb_optim = nu.init_from_checkpoint(checkpoint, inv_bb_module, neptune_params)
        # detr, detr_optim = nu.init_from_checkpoint(checkpoint, detr_module, neptune_params)
        # inv_enc, inv_enc_optim = nu.init_from_checkpoint(checkpoint, inv_enc_module, neptune_params)
        # best_loss = checkpoint['best_loss']
        # test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        # last_epoch = neptune_params['epochs']
        #
        # optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
        # models = [inv_bb, inv_enc, inv_dec, detr]
        # run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
        #                        monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        raise NotImplementedError()
    else:
        vit = timm.create_model(neptune_params['init_vit'], pretrained=True)

        if inv_bb_id is not None:
            inv_bb = nu.init_model_from_neptune(run_id=inv_bb_id, module=inv_bb_module)
        else:
            inv_bb = nu.init_model_from_params(module=inv_bb_module, params=neptune_params)

        if inv_enc_id is not None:
            inv_enc = nu.init_model_from_neptune(run_id=inv_enc_id, module=inv_enc_module)
        else:
            inv_enc = nu.init_model_from_params(params=neptune_params, module=inv_enc_module)

        inv_bb.to(config.DEVICE), inv_enc.to(config.DEVICE), vit.to(config.DEVICE)

        best_inv_bb = copy.deepcopy(inv_bb)
        best_inv_enc = copy.deepcopy(inv_enc)

        models = [inv_bb, inv_enc]

        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                               capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

        run['params'] = neptune_params
        test_step, train_step, last_epoch = 0, -1, 0
        inv_bb_optim = create_optimizer_v2(model_or_params=inv_bb, lr=optim_configs['inv_vit_bb']['lr'],
                                           opt=optim_configs['inv_vit_bb']['module_type'])
        inv_enc_optim = create_optimizer_v2(model_or_params=inv_enc, lr=optim_configs['inv_vit_encoder']['lr'],
                                            opt=optim_configs['inv_vit_encoder']['module_type'])

        optims = [inv_bb_optim, inv_enc_optim]

        best_bb, best_enc = test_train_in_parallel(vit=vit, inv_bb=inv_bb, inv_enc=inv_enc, dataloader=dataloader_val, run=run, normalize=normalize)
        test_train_chain_losses(inv_enc=best_inv_enc, inv_bb=best_inv_bb, run=run, vit=vit, dataloader=dataloader_val, normalize=normalize)
        nu.upload_model_states(models=models, run=run)
        nu.upload_checkpoint(models=models, optims=optims, best_loss=(best_bb, best_enc), run=run,
                             test_step=test_step, train_step=train_step)

    bb_scaler = NativeScaler()
    enc_scaler = NativeScaler()

    for epoch in range(epochs):
        for batch_id, (img, _) in enumerate(dataloader_train):
            img = img.to(config.DEVICE)
            with torch.no_grad():
                bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
                enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)
                # vit_out = vu.enc_emb_to_vit_out(enc_emb=enc_emb, vit=vit)
            img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
            if normalize:
                img_recon = cu.normalize(img_recon)
            bb_loss = F.mse_loss(input=img_recon, target=img)
            inv_bb_optim.zero_grad()
            bb_scaler(bb_loss, inv_bb_optim, clip_grad=1.0, parameters=inv_bb.parameters())
            run["train/bb_loss"].append(bb_loss.item())

            bb_emb_recon = vu.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc)
            enc_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
            inv_enc_optim.zero_grad()
            enc_scaler(enc_loss, inv_enc_optim, clip_grad=1.0, parameters=inv_enc.parameters())
            run["train/enc_loss"].append(enc_loss.item())

            train_step += 1
        bb_val, enc_val = test_train_in_parallel(vit, inv_bb, inv_enc, dataloader_val, run=run, normalize=normalize)

        test_step += 1
        if bb_val < best_bb:
            nu.upload_model_state(model=inv_bb, run=run)
            best_bb = bb_val
            best_inv_bb = copy.deepcopy(inv_bb)
        if enc_val < best_enc:
            nu.upload_model_state(model=inv_enc, run=run)
            best_enc = enc_val
            best_inv_enc = copy.deepcopy(inv_enc)

        test_train_chain_losses(inv_enc=best_inv_enc, inv_bb=best_inv_bb, run=run, vit=vit, dataloader=dataloader_val, normalize=normalize)

        nu.upload_checkpoint(models=models, optims=optims, best_loss=(best_bb, best_enc),
                             run=run, test_step=test_step, train_step=train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
