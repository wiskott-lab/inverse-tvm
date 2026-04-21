import torch
import config
from modules.inv_detr_bb import models as inverse_backbone_module
from modules.inv_detr_bb.utils import test_inv_bb
from modules.detr.hubconf import detr_resnet50
import argparse
from torch.functional import F
from pathlib import Path
import uuid
import neptune
from tools import detr_utils as du, training_utils as tu, neptune_utils as nu, coco_utils as cu
from tools.misc_utils import get_parent_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id run', type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=3e-3)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=32)
    parser.add_argument("--sched", type=str, default="linearW")

    args = parser.parse_args()

    epochs = args.epochs
    lr = args.learning_rate
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = tu.init_seeds(None)
    model_path = config.TMP_DIR / (str(uuid.uuid4()))  # tmp path for storing model state dict
    best_loss = 2147483647

    model_configs = {'inv_detr_bb': {'module_type': 'LinearDecoderEnhanced'}}
    optim_configs = {'inv_detr_bb': {'module_type': 'Adam', 'lr': lr}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'COCO2017', 'model_configs': model_configs, 'optim_configs': optim_configs,
                      'detr': 'resnet50'}

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')
    source_files = [str(Path(__file__))]

    detr = detr_resnet50(pretrained=True)
    backbone = detr.backbone
    input_proj = detr.input_proj
    backbone.to(config.DEVICE)
    input_proj.to(config.DEVICE)

    if run_id:
        run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        checkpoint = nu.get_checkpoint(run_id=run_id)
        neptune_params = nu.get_params(run_id=run_id)
        inv_bb, inv_bb_optim = nu.init_from_checkpoint(checkpoint, inverse_backbone_module, neptune_params)
        best_loss = checkpoint['best_loss']
        test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        last_epoch = neptune_params['epochs']
        inv_bb.to(config.DEVICE)
    else:
        inv_bb = nu.init_model_from_params(module=inverse_backbone_module, params=neptune_params)
        inv_bb_optim = nu.init_optim_from_params(inv_bb, neptune_params)
        run = neptune.init_run(project=config.PROJECT, source_files=source_files, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        run['params'] = neptune_params
        inv_bb.to(config.DEVICE)
        best_loss = test_inv_bb(inv_bb=inv_bb, run=run, detr=detr, dataloader=dataloader_test)
        test_step, train_step, last_epoch = 0, -1, 0
        nu.upload_model_state(model=inv_bb, run=run)
        nu.upload_checkpoint(models=inv_bb, optims=inv_bb_optim, best_loss=best_loss, run=run,test_step=test_step,
                             train_step=train_step)

    for epoch in range(epochs):
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                bb_emb, _, _ = du.nested_tensor_to_bb_emb(nested_tensor, detr)
            recon = cu.normalize(du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=nested_tensor.tensors)  # may need to account for resizing here
            run["train/loss"].append(loss)
            tu.optim_step(inv_bb_optim, loss)
            train_step += 1
        loss = test_inv_bb(inv_bb=inv_bb, run=run, detr=detr, dataloader=dataloader_test)
        test_step += 1
        if loss < best_loss:
            nu.upload_model_state(inv_bb, run)
            best_loss = loss
        nu.upload_checkpoint(inv_bb, inv_bb_optim, best_loss, run, test_step, train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
