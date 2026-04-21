import timm
from neptune.utils import stringify_unsupported
from torchvision import transforms, datasets
import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.inv_resnet.utils import test_inverse_models
from tools import training_utils
import tools.neptune_utils as nu
from modules.inv_resnet import models as inv_resnet_module
from torch.utils.data import DataLoader
import tools.resnet_utils as ru
from tools.misc_utils import get_parent_file
from timm.optim import create_optimizer_v2
from timm.utils.cuda import NativeScaler
import tools.coco_utils as cu
from modules.detr.hubconf import detr_resnet50



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default=None)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=1000)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=64)
    parser.add_argument("--resnet_id", '-ri', type=str, default='default')

    # parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default='OB-333')
    # parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default='OB-320')
    # parser.add_argument("--inv_dec_id", "-idid", help='neptune id of inv dec', type=str, default='OB-325')

    args = parser.parse_args()
    epochs = args.epochs

    lr = args.lr
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device


    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = training_utils.init_seeds(None)

    model_configs = {
        '0': {'module_type': 'InverseResnetBlock', 'in_channels': 64, 'out_channels': 3, 'last_output': True},
        '1': {'module_type': 'InverseResnetBlock', 'in_channels': 256, 'out_channels': 64, 'upsample': False},
        '2': {'module_type': 'InverseResnetBlock', 'in_channels': 512},
        '3': {'module_type': 'InverseResnetBlock', 'in_channels': 1024},
        '4': {'module_type': 'InverseResnetBlock', 'in_channels': 2048}}

    optim_configs = {'0': {'opt': 'Adam', 'lr': 0.0001},
                     '1': {'opt': 'Adam', 'lr': 0.0001},
                     '2': {'opt': 'Adam', 'lr': 0.0001},
                     '3': {'opt': 'Adam', 'lr': 0.0001},
                     '4': {'opt': 'Adam', 'lr': 0.0001}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'imagenet1k', 'model_configs': model_configs, 'optim_configs': optim_configs,
                      'resnet_id': args.resnet_id}

    #
    # transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
    #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #
    # dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
    #                               num_workers=4, shuffle=True)
    #
    # dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    # dataloader_val = DataLoader(dataset_val, batch_size=int(args.batch_size), drop_last=False, pin_memory=True,
    #                             num_workers=4, shuffle=False)

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_val = cu.get_dataloader(batch_size=batch_size, dataset_type='val')

    source_files = [str(Path(__file__))]

    if args.resnet_id == 'default':
        forward_model = timm.create_model('resnet50', pretrained=True)
    else:
        raise NameError('Unknown bb')

    # from modules.detr.util.misc import NestedTensor
    #
    # a = NestedTensor(torch.rand(4, 3, 640, 480), None)
    # g = detr.backbone(a)
    forward_model.to(config.DEVICE)
    forward_model.eval()
    inverse_models, optims = [], []
    module_keys = sorted(list(model_configs.keys()), key=int)
    for key in module_keys:
        model = nu.init_module(module=inv_resnet_module, **model_configs[key])
        model.to(config.DEVICE).eval()
        optim = create_optimizer_v2(model_or_params=model, **optim_configs[key])
        inverse_models.append(model)
        optims.append(optim)

    run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                           capture_hardware_metrics=False,
                           monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

    run['params'] = stringify_unsupported(neptune_params)
    test_step, train_step, last_epoch = 0, -1, 0

    best_losses = test_inverse_models(inverse_models=inverse_models, dataloader=dataloader_val, forward_model=forward_model, run=run)

    nu.upload_model_state_keys(models=inverse_models, run=run)
    nu.upload_checkpoint_keys(models=inverse_models, optims=optims, best_loss=best_losses.tolist(), run=run,
                              test_step=test_step, train_step=train_step, keys=('0', '1', '2', '3', '4'))

    optim_scalers = [NativeScaler() for optim in optims]

    for epoch in range(epochs):
        for model in inverse_models:
            model.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            img = nested_tensor.tensors.to(config.DEVICE)
            with torch.no_grad():
                embs = ru.tensor_to_embs(model=forward_model, tensor=img)
            recon_embs = ru.invert_embs(embs=embs, inv_networks=inverse_models)
            for i in range(len(recon_embs)):
                if i == 0:
                    loss = F.mse_loss(input=cu.normalize(recon_embs[i]), target=embs[i])
                else:
                    loss = F.mse_loss(input=recon_embs[i], target=embs[i])
                optims[i].zero_grad()
                optim_scalers[i](loss, optims[i], clip_grad=1.0, parameters=inverse_models[i].parameters())
                run[f"train/{str(i)}"].append(loss.item())
            train_step += 1
        eval_value = test_inverse_models(inverse_models=inverse_models, dataloader=dataloader_val, forward_model=forward_model, run=run)
        test_step += 1
        for i in range(len(eval_value)):
            if eval_value[i] < best_losses[i]:
                best_losses[i] = eval_value[i]
                nu.upload_model_state_key(model=inverse_models[i], run=run, key=str(i))
        nu.upload_checkpoint_keys(models=inverse_models, optims=optims, best_loss=best_losses.tolist(), run=run,
                                  test_step=test_step, train_step=train_step, keys=('0', '1', '2', '3', '4'))
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
