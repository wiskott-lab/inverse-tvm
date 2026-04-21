from modules.finetuned_vit.utils import test_classic_in_par
import tools.neptune_utils as nu
import tools.coco_utils as cu

from torchvision import transforms, datasets

import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from tools import training_utils
from modules.inv_vit_bb import models as inv_bb_module
from modules.inv_vit_enc import models as inv_enc_module
from torch.utils.data import DataLoader
from tools.misc_utils import get_parent_file
from timm.optim import create_optimizer_v2
from timm.utils.cuda import NativeScaler
import tools.vit_utils as vu

import timm


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

    transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])



    source_files = [str(Path(__file__))]

    train_bb_to_img, train_enc_to_img = True, True
    bb_to_img_optim, enc_to_img_optim = None, None
    bb_to_img, enc_to_img = None, None

    init_bb_id, init_enc_id = None, None

    bb_ids = (init_bb_id,)
    bb_modules = (inv_bb_module,)

    enc_ids = (init_bb_id, init_enc_id)
    enc_modules = (inv_bb_module, inv_enc_module)


    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'optim_configs': optim_configs, 'model_configs': model_configs,
                      'init_bb_id': init_bb_id, 'init_enc_id': init_enc_id,  'init_vit': 'deit3_base_patch16_224'}  # deit3_base_patch16_224, vit_base_patch16_224}

    if neptune_params['init_vit'] == 'deit3_base_patch16_224':
        transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        normalize = True
    else:
        transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
        transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
        normalize = False

    dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
                                  num_workers=4, shuffle=True)

    dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=int(4 * args.batch_size), drop_last=False, pin_memory=True,
                                num_workers=4, shuffle=False)


    if run_id:
        raise NotImplementedError()
        # neptune_params = nu.get_params(run_id=run_id)
        # checkpoint = nu.get_checkpoint(run_id=run_id)
        # inv_bb, inv_bb_optim = nu.init_from_checkpoint(checkpoint, inv_bb_module, neptune_params)
        # detr, detr_optim = nu.init_from_checkpoint(checkpoint, detr_module, neptune_params)
        # inv_enc, inv_enc_optim = nu.init_from_checkpoint(checkpoint, inv_enc_module, neptune_params)
        # inv_dec, inv_dec_optim = nu.init_from_checkpoint(checkpoint, inv_dec_module, neptune_params)
        # best_loss = checkpoint['best_loss']
        # test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        # last_epoch = neptune_params['epochs']
        #
        # optims = [inv_bb_optim, inv_enc_optim, inv_dec_optim, detr_optim]
        # models = [inv_bb, inv_enc, inv_dec, detr]
        # run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
        #                        monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
    else:
        vit = timm.create_model(neptune_params['init_vit'], pretrained=True)
        vit.to(config.DEVICE)
        bb_to_img = init_submodules(params=neptune_params, run_ids=bb_ids, modules=bb_modules)
        for module in bb_to_img:
            module.to(config.DEVICE)
        bb_to_img_params = bb_to_img[0].parameters()
        bb_to_img_optim = create_optimizer_v2(model_or_params=bb_to_img_params,
                                              lr=optim_configs['inv_vit_bb']['lr'],
                                              opt=optim_configs['inv_vit_bb']['module_type'])
        enc_to_img = init_submodules(params=neptune_params, run_ids=enc_ids, modules=enc_modules)
        for module in enc_to_img:
            module.to(config.DEVICE)
        enc_to_img_params = list(enc_to_img[0].parameters()) + list(enc_to_img[1].parameters())
        enc_to_img_optim = create_optimizer_v2(model_or_params=enc_to_img_params,
                                               lr=optim_configs['inv_vit_encoder']['lr'],
                                               opt=optim_configs['inv_vit_encoder']['module_type'])

        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                               capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

        run['params'] = neptune_params
        test_step, train_step, last_epoch = 0, -1, 0

        optims = [bb_to_img_optim, enc_to_img_optim]
        models = [bb_to_img, enc_to_img]

        best_bb, best_enc = test_classic_in_par(vit=vit, bb_to_img=bb_to_img, enc_to_img=enc_to_img,
                                                dataloader=dataloader_val, run=run, normalize=normalize)
        nu.upload_model_state_tuple(models=bb_to_img, run=run, model_id='bb')
        nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')

        nu.upload_checkpoint_classic_vit(models=models, optims=optims, best_loss=(best_bb, best_enc),
                                     run=run, test_step=test_step, train_step=train_step)

    bb_scaler = NativeScaler()
    enc_scaler = NativeScaler()

    for epoch in range(epochs):
        for batch_id, (img, _) in enumerate(dataloader_train):
            img = img.to(config.DEVICE)
            with torch.no_grad():
                bb_emb = vu.tensor_to_bb_emb(tensor=img, vit=vit)
                enc_emb = vu.bb_emb_to_enc_emb(bb_emb=bb_emb, vit=vit)
            if train_bb_to_img:
                bb_to_img_recon = vu.bb_emb_to_tensor(bb_emb=bb_emb, inv_bb=bb_to_img[0])
                if normalize:
                    bb_loss = F.mse_loss(input=cu.normalize(bb_to_img_recon), target=img)
                else:
                    bb_loss = F.mse_loss(input=bb_to_img_recon, target=img)
                bb_to_img_optim.zero_grad()
                bb_scaler(bb_loss, bb_to_img_optim, clip_grad=1.0, parameters=bb_to_img_params)
                run["train/bb_loss"].append(bb_loss.item())
            if train_enc_to_img:
                enc_to_img_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_bb=enc_to_img[0],
                                                        inv_enc=enc_to_img[1])
                if normalize:
                    enc_loss = F.mse_loss(input=cu.normalize(enc_to_img_recon), target=img)
                else:
                    enc_loss = F.mse_loss(input=enc_to_img_recon, target=img)
                enc_to_img_optim.zero_grad()
                enc_scaler(enc_loss, enc_to_img_optim, clip_grad=1.0, parameters=enc_to_img_params)
                run["train/enc_loss"].append(enc_loss.item())

            train_step += 1
        bb_val, enc_val = test_classic_in_par(vit=vit, bb_to_img=bb_to_img, enc_to_img=enc_to_img,
                                                dataloader=dataloader_val, run=run, normalize=normalize)
        test_step += 1
        if train_bb_to_img and bb_val < best_bb:
            nu.upload_model_state_tuple(models=bb_to_img, run=run, model_id='bb')
            best_bb = bb_val
        if train_enc_to_img and enc_val < best_enc:
            nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')
            best_enc = enc_val

        nu.upload_checkpoint_classic_vit(models=models, optims=optims, best_loss=(best_bb, best_enc),
                                     run=run, test_step=test_step, train_step=train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
