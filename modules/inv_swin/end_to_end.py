import timm
from torchvision import transforms, datasets
import torch
import config
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.inv_swin.utils import test_inverse_swin_end
from tools import training_utils
import tools.neptune_utils as nu
from modules.inv_swin import models as inv_swin_module
from torch.utils.data import DataLoader
import tools.swin_utils as su
from tools.misc_utils import get_parent_file
from timm.optim import create_optimizer_v2
from timm.utils.cuda import NativeScaler
import tools.coco_utils as cu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id of checkpoint', type=str, default=None)
    parser.add_argument("--lr", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=1000)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

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
    pretrained_resnet = True

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

    optim_config = {'opt': 'Adam', 'lr': 0.0001}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'imagenet1k', 'model_configs': model_configs, 'optim_configs': optim_config}

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

    source_files = [str(Path(__file__))]

    swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    swin.to(config.DEVICE)
    swin.eval()
    models = []
    module_keys = sorted(list(model_configs.keys()), key=int)
    for key in module_keys:
        model = nu.init_module(module=inv_swin_module, **model_configs[key])
        model.to(config.DEVICE).eval()
        models.append(model)
    parameters = list(models[0].parameters()) + list(models[1].parameters()) + list(models[2].parameters()) + list(
        models[3].parameters()) + list(models[4].parameters())
    optim = create_optimizer_v2(model_or_params=parameters, **optim_config)

    run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
                           capture_hardware_metrics=False,
                           monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
    run['params'] = neptune_params
    test_step, train_step, last_epoch = 0, -1, 0

    best_loss = test_inverse_swin_end(models=models, dataloader=dataloader_val, swin=swin, run=run)

    nu.upload_model_state_keys(models=models, run=run)
    nu.upload_checkpoint_keys(models=models, optims=optim, best_loss=best_loss, run=run, test_step=test_step,
                              train_step=train_step, keys=('0', '1', '2', '3', '4'))

    optim_scaler = NativeScaler()

    for epoch in range(epochs):
        for batch_id, (img, _) in enumerate(dataloader_train):
            img = img.to(config.DEVICE)
            with torch.no_grad():
                emb = su.tensor_to_embs(tensor=img, model=swin)[-2]
            recon = su.chain_invert(emb=emb, inv_networks=models)
            loss = F.mse_loss(input=cu.normalize(recon), target=img)
            optim.zero_grad()
            optim_scaler(loss, optim, clip_grad=1.0, parameters=parameters)
            run[f"train/recon_loss"].append(loss.item())
            train_step += 1
        eval_value = test_inverse_swin_end(models=models, dataloader=dataloader_val, swin=swin, run=run)
        test_step += 1
        if eval_value < best_loss:
            best_loss = eval_value
            nu.upload_model_state_keys(models=models, run=run)
        nu.upload_checkpoint_keys(models=models, optims=optim, best_loss=best_loss, run=run,
                                  test_step=test_step, train_step=train_step, keys=('0', '1', '2', '3', '4'))
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
