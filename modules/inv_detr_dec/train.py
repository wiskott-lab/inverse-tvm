import torch
import config
from modules.detr.hubconf import detr_resnet50
import tools.coco_utils as cu
import argparse
from torch.functional import F
from pathlib import Path
import neptune
from modules.inv_detr_dec.utils import test_inv_dec
from modules.inv_detr_dec import models as inv_dec_module
from tools.misc_utils import get_parent_file
from tools import training_utils, detr_utils
from tools import detr_utils as du
from tools import neptune_utils as nu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id run', type=str, default=None)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)

    args = parser.parse_args()
    epochs = args.epochs
    run_id = args.run_id
    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = training_utils.init_seeds(None)
    model_configs = {'inv_dec': {'module_type': 'InverseTransformerDecoder'}}
    optim_configs = {'inv_dec': {'module_type': 'Adam', 'lr': 0.0001}}

    params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': 128, 'seed': seed,
              'dataset_id': 'COCO2017', 'model_configs': model_configs, 'optim_configs': optim_configs,
              'detr': 'detr_resnet50'}

    dataloader_train = cu.get_dataloader(batch_size=params['batch_size'], dataset_type='train')
    dataloader_val = cu.get_dataloader(batch_size=params['batch_size'], dataset_type='val')

    detr = detr_resnet50(pretrained=True)
    detr.to(config.DEVICE).eval()

    if run_id:
        run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
                               monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
        checkpoint = nu.get_checkpoint(run_id=run_id)
        params = nu.get_params(run_id=run_id)
        inv_dec, inv_dec_optim = nu.init_from_checkpoint(checkpoint, inv_dec_module,
                                                         params)
        best_loss = checkpoint['best_loss']
        test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
        last_epoch = params['epochs']
        train_detr_in_eval_mode = params['train_detr_in_eval_mode']
        train_inv_dec_in_eval_mode = params['train_inv_dec_in_eval_mode']
        inv_dec.to(config.DEVICE)
    else:
        inv_dec = nu.init_model_from_params(module=inv_dec_module, params=params)
        inv_dec_optim = nu.init_optim_from_params(inv_dec, params)

        run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))], capture_stderr=False,
                               capture_hardware_metrics=False, monitoring_namespace='monitoring', capture_stdout=False)
        run['params'] = params

        inv_dec.to(config.DEVICE)
        best_loss = test_inv_dec(inv_dec=inv_dec, run=run, detr=detr, dataloader=dataloader_val)
        test_step, train_step, last_epoch = 0, -1, 0
        nu.upload_model_state(model=inv_dec, run=run)
        nu.upload_checkpoint(models=inv_dec, optims=inv_dec_optim, best_loss=best_loss, run=run,
                             test_step=test_step, train_step=train_step)

    for epoch in range(epochs):
        detr.train(), inv_dec.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                enc_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, pos=pos, mask=mask)
            recon = detr_utils.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
            loss = F.mse_loss(input=recon, target=enc_emb)
            run["train/loss"].append(loss)
            training_utils.optim_step(inv_dec_optim, loss)
            train_step += 1
        loss = test_inv_dec(inv_dec=inv_dec, run=run, detr=detr, dataloader=dataloader_val)
        test_step += 1
        if loss < best_loss:
            nu.upload_model_state(inv_dec, run)
            best_loss = loss
        nu.upload_checkpoint(inv_dec, inv_dec_optim, best_loss, run, test_step, train_step)
        run['params']['epochs'] = last_epoch + epoch + 1
    run.stop()
