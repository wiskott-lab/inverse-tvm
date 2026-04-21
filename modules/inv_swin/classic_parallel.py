from torch.utils.checkpoint import checkpoint

from modules.inv_swin.utils import test_classic_in_parallel
import tools.neptune_utils as nu
import tools.coco_utils as cu

from torchvision import transforms, datasets
import os
import wandb
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
import tools.swin_utils as su
from modules.inv_swin import models as inv_swin_module
import timm

# os.environ["WANDB_DISABLE_SYSTEM_STATS"] = "true"


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
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default='ubws4pxp')
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=1000)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

    # parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default='OB-333')
    # parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default='OB-320')
    # parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default='OB-325')
    # parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default=None)
    # parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default=None)

    args = parser.parse_args()
    epochs = args.epochs

    lr = args.lr
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    # inv_bb_id = args.inv_bb_id
    # inv_enc_id = args.inv_enc_id

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = training_utils.init_seeds(None)

    model_configs = {
        '0': {'module_type': 'SwinBackbone'},
        '1': {'module_type': 'InverseSwinTransformerStage', 'dim': 128, 'out_dim': 128,
              'input_resolution': (56, 56), 'output_resolution': (56, 56), 'depth': 2, 'upsample': False,
              'num_heads': 4},
        '2': {'module_type': 'InverseSwinTransformerStage', 'dim': 256, 'out_dim': 128,
              'input_resolution': (28, 28), 'output_resolution': (56, 56), 'depth': 2, 'upsample': True,
              'num_heads': 8},
        '3': {'module_type': 'InverseSwinTransformerStage', 'dim': 512, 'out_dim': 256,
              'input_resolution': (14, 14), 'output_resolution': (28, 28), 'depth': 18, 'upsample': True,
              'num_heads': 16},
        '4': {'module_type': 'InverseSwinTransformerStage', 'dim': 1024, 'out_dim': 512,
              'input_resolution': (7, 7), 'output_resolution': (14, 14), 'depth': 2, 'upsample': True,
              'num_heads': 32}}

    optim_configs = {'0': {'module_type': 'adam', 'lr': 0.001},
                     '1': {'module_type': 'adam', 'lr': 0.001},
                     '2': {'module_type': 'adam', 'lr': 0.001},
                     '3': {'module_type': 'adam', 'lr': 0.0001},
                     '4': {'module_type': 'adam', 'lr': 0.0001}}


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
                      'dataset_id': 'imgnet', 'optim_configs': optim_configs, 'model_configs': model_configs}

    transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
                                  num_workers=4, shuffle=True)

    dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=int(args.batch_size), drop_last=False, pin_memory=True,
                                num_workers=4, shuffle=False)

    if run_id:
        swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        swin.to(config.DEVICE)
        best_losses = torch.zeros(5).to(config.DEVICE)
        module_keys = sorted(list(model_configs.keys()), key=int)
        stage_1_to_0, stage_2_to_0, stage_3_to_0, stage_4_to_0, stage_5_to_0 = [], [], [], [], []
        all_models = [stage_1_to_0, stage_2_to_0, stage_3_to_0, stage_4_to_0, stage_5_to_0]
        all_params = [[] for _ in range(5)]
        all_optims = []
        all_scalers = [NativeScaler() for _ in range(5)]
        for i in range(5):
            ckpt = torch.load(config.RUNS_DIR / run_id / str(i), map_location='cpu')
            for j in range(i + 1):
                model = nu.init_module(module=inv_swin_module, **model_configs[module_keys[j]])
                model.load_state_dict(ckpt['model_states'][str(i - j)])
                model.to(config.DEVICE).eval()
                all_models[i].insert(0, model)
                all_params[i].extend(list(model.parameters()))
            all_optims.append(create_optimizer_v2(model_or_params=all_params[i],
                                                  lr=optim_configs[str(i)]['lr'],
                                                  opt=optim_configs[str(i)]['module_type']))
            all_optims[i].load_state_dict(ckpt['optim_state'])

            best_losses[i] = ckpt['best_loss']
            train_step = ckpt['train_step']
            val_step = ckpt['val_step']
    else:
        swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        swin.to(config.DEVICE)
        module_keys = sorted(list(model_configs.keys()), key=int)
        stage_1_to_0, stage_2_to_0, stage_3_to_0, stage_4_to_0, stage_5_to_0 = [], [], [], [], []
        all_models = [stage_1_to_0, stage_2_to_0, stage_3_to_0, stage_4_to_0, stage_5_to_0]
        all_params = [[] for _ in range(5)]
        all_optims = []
        all_scalers = [NativeScaler() for _ in range(5)]

        for i in range(len(all_models)):
            for j in range(i + 1):
                model = nu.init_module(module=inv_swin_module, **model_configs[module_keys[j]])
                model.to(config.DEVICE).eval()
                all_models[i].insert(0, model)
                all_params[i].extend(list(model.parameters()))
            all_optims.append(create_optimizer_v2(model_or_params=all_params[i],
                                                    lr=optim_configs[str(i)]['lr'],
                                                    opt=optim_configs[str(i)]['module_type']))

        val_step, train_step  = 0, 0

        best_losses = test_classic_in_parallel(all_models=all_models, dataloader=dataloader_val, swin=swin,
                                               step=val_step)
        # for i in range(len(all_models)):
        #     nu.save_model_state_tuple(models=all_models[i], run=run, model_id=str(i))
        #
        # for i in range(len(all_models)):
        #     nu.save_checkpoint_tuple(models=all_models[i], optims=all_optims[i], run=run, best_loss=best_losses[i],
        #                              model_state_key=str(i), train_step=train_step, val_step=val_step)

    with wandb.init(project=config.PROJECT, config=neptune_params, settings=wandb.Settings(_disable_stats=True),
                    id=run_id, resume='must') as run:
        nu.prepare_run(run, neptune_params)
        # training metrics use train/step


        # nu.upload_checkpoint_classic(models=models, optims=optims, best_loss=(best_bb, best_enc, best_dec, best_detect),
        #                              run=run, test_step=test_step, train_step=train_step)
        # nu.upload_model_state_tuple(models=bb_to_img, run=run, model_id='bb')
        # nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')

        # nu.upload_checkpoint_classic_vit(models=models, optims=optims, best_loss=(best_bb, best_enc),
        #                                  run=run, test_step=test_step, train_step=train_step)
        for epoch in range(epochs):
            for batch_id, (img, _) in enumerate(dataloader_train):
                img = img.to(config.DEVICE)
                with torch.no_grad():
                    int_embs = su.tensor_to_embs(tensor=img, model=swin)
                for i in range(len(all_models)):
                    current_model = all_models[i]
                    current_emb = int_embs[i + 1]
                    for j in range(len(current_model)):
                        current_emb = current_model[j](current_emb)

                    loss = F.mse_loss(input=cu.normalize(current_emb), target=img)
                    all_optims[i].zero_grad()
                    all_scalers[i](loss, all_optims[i], clip_grad=1.0, parameters=all_params[i])
                    wandb.log({"train/step": train_step, f'train/loss_{i}': loss})
                train_step += 1
            val_step += 1
            val_losses = test_classic_in_parallel(all_models=all_models, dataloader=dataloader_val, swin=swin, step=val_step)
            for i in range(len(all_models)):
                if val_losses[i] < best_losses[i]:
                    nu.save_model_state_tuple(models=all_models[i], run=run, model_id=str(i))
                    best_losses[i] = val_losses[i]
            for i in range(len(all_models)):
                nu.save_checkpoint_tuple(models=all_models[i], optims=all_optims[i], run=run, best_loss=best_losses[i],
                                         model_state_key=str(i), train_step=train_step, val_step=val_step)





   #     if train_bb_to_img and bb_val < best_bb:
    #         nu.upload_model_state_tuple(models=bb_to_img, run=run, model_id='bb')
    #         best_bb = bb_val
    #     if train_enc_to_img and enc_val < best_enc:
    #         nu.upload_model_state_tuple(models=enc_to_img, run=run, model_id='enc')
    #         best_enc = enc_val
    #
    #     nu.upload_checkpoint_classic_vit(models=models, optims=optims, best_loss=(best_bb, best_enc),
    #                                      run=run, test_step=test_step, train_step=train_step)
    #     run['params']['epochs'] = last_epoch + epoch + 1
    # run.stop()
