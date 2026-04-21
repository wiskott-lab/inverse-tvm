import torch
import config
import tools.training_utils as tu
import tools.coco_utils as cu
import argparse
from torch.functional import F
from pathlib import Path
import neptune
import tools.neptune_utils as nu
from modules.inv_vit_enc.utils import test_inv_enc
from modules.inv_vit_enc import models as inv_vit_enc_module
import tools.vit_utils as vu
from tools.misc_utils import get_parent_file
import timm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id run', type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = tu.init_seeds(None)
    model_configs = {'inv_vit_enc': {'module_type': 'InverseViTEncoder'}}
    optim_configs = {'inv_vit_enc': {'module_type': 'Adam', 'lr': lr}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'model_configs': model_configs, 'optim_configs': optim_configs,
                      'detr': 'resnet50'}

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train', transform=cu.vit_transforms)
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='val', transform=cu.vit_transforms)

    vit = timm.create_model("vit_base_patch16_224" , pretrained=True)
    vit.to(config.DEVICE)

    if run_id:
        run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        checkpoint = nu.get_checkpoint(run_id=run_id)
        neptune_params = nu.get_params(run_id=run_id)
        inv_enc, inv_enc_optim = nu.init_from_checkpoint(checkpoint, inv_vit_enc_module, neptune_params)
        best_loss = checkpoint['best_loss']
        test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        last_epoch = neptune_params['epochs']
        inv_enc.to(config.DEVICE)
    else:
        inv_enc = nu.init_model_from_params(module=inv_vit_enc_module, params=neptune_params)
        inv_enc_optim = nu.init_optim_from_params(inv_enc, neptune_params)
        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))], capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        best_loss = test_inv_enc(inv_enc=inv_enc, run=run, vit=vit, dataloader=dataloader_test)
        run['params'] = neptune_params
        inv_enc.to(config.DEVICE)
        test_step, train_step, last_epoch = 0, -1, 0
        nu.upload_model_state(model=inv_enc, run=run)
        nu.upload_checkpoint(models=inv_enc, optims=inv_enc_optim, best_loss=best_loss, run=run, test_step=test_step,
                             train_step=train_step)
    for epoch in range(epochs):
        vit.train(), inv_enc.train()
        for batch_id, (tensor, _) in enumerate(dataloader_train):
            tensor = tensor.tensors.to(config.DEVICE)
            with torch.no_grad():
                input_emb = vu.tensor_to_bb_emb(tensor, vit)
                enc_emb = vu.tensor_to_enc_emb(tensor, vit)
            recon = vu.enc_emb_to_bb_emb(enc_emb, inv_enc)
            loss = F.mse_loss(input=recon, target=input_emb)
            run["train/loss"].append(loss)
            tu.optim_step(inv_enc_optim, loss)
            train_step += 1
        loss = test_inv_enc(inv_enc=inv_enc, run=run, vit=vit, dataloader=dataloader_test)
        test_step += 1
        if loss < best_loss:
            nu.upload_model_state(inv_enc, run)
            best_loss = loss
        nu.upload_checkpoint(inv_enc, inv_enc_optim, best_loss, run, test_step, train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
