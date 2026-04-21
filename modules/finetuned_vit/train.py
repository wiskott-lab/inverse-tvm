import torch
import config
import tools.training_utils as tu
import tools.coco_utils as cu
import argparse
from torch.functional import F
from pathlib import Path
import neptune
import tools.neptune_utils as nu
from modules.finetuned_vit.utils import test_finetuned_vit
from modules.inv_vit_bb import models as inv_bb_module
from modules.inv_vit_enc import models as inv_enc_module
import timm.models.vision_transformer as vit_module
from torch.utils.data import DataLoader
import tools.vit_utils as vu
from tools.misc_utils import get_parent_file
from torchvision import transforms, datasets
from timm.utils.cuda import NativeScaler
from timm.optim import create_optimizer_v2
import wandb
# STEP_FROM_ENC = 'enc_emb'

TRAINING_MODE_CHAIN = 'chain'
TRAINING_MODE_SUM = 'sum'
TRAINING_MODE_ISOLATED = 'isolated'
TRAINING_MODE_BACKWARDS_ONLY = 'backwards_only'


def _zero_grad_optims():
    for optim in optims:
        optim.zero_grad()


# def _clip_grads():
#     for model in models:
#         for param in model.parameters():
#             if param.grad is not None:
#                 param.grad = param.grad.clamp(-1, 1)  # Creates a new tensor instead of modifying in-place


def _clip_grads():
    for model in models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


# def _backwards_only_step():

#
# def _chain_step():
#     chain_recon = du.dec_emb_to_img(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc,
#                                     inv_bb=inv_bb)
#     chain_recon_loss = F.mse_loss(input=du.normalize(chain_recon), target=nested_tensor.tensors)
#     loss = (1 - trade_off) * detr_loss + trade_off * chain_recon_loss
#     _zero_grad_optims()
#     loss.backward(retain_graph=True)
#     _clip_grads()
#     for optim in optims:
#         optim.step()
#     run["train/loss"].append(loss)
#     run["train/recon_loss"].append(chain_recon_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--run_id", "-r", help='neptune id run', type=str, default=None)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=512)

    parser.add_argument("--learning_rate_bb", "-lr-back", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_enc", "-lr-enc", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_vit", "-lr-custom_vit", help='learning rate', type=float, default=0.00001)

    parser.add_argument("--inv_bb_id", "-ibid", help='neptune id of inv bb', type=str, default=None)
    parser.add_argument("--inv_enc_id", "-ieid", help='neptune id of inv enc', type=str, default=None)
    parser.add_argument("--vit_id", "-vid", help='neptune id of inv bb', type=str, default='vit_base_patch16_224')
    parser.add_argument("--trade_off", "-t", help='trade-off', type=float, default=1.0)

    parser.add_argument("--training_mode", "-tm", help='training mode', type=str, default=TRAINING_MODE_CHAIN)
    # parser.add_argument("--step_from", "-s", help='step_from', type=str, default=STEP_FROM_ENC)
    parser.add_argument("--eval_loss_id", "-eid", help='eval_loss_id', type=str, default='enc_emb')

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    run_id = args.run_id
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    lr_bb = args.learning_rate_bb
    lr_enc = args.learning_rate_enc
    lr_vit = args.learning_rate_vit

    inv_bb_id = args.inv_bb_id
    inv_enc_id = args.inv_enc_id
    vit_id = args.vit_id

    trade_off = args.trade_off

    training_mode = args.training_mode
    # step_from = args.step_from
    eval_loss_id = args.eval_loss_id

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = tu.init_seeds(None)
    model_configs = {'inv_vit_bb': {'as_in': inv_bb_id},
                     'inv_vit_encoder': {'as_in': inv_enc_id},
                     'vision_transformer': {'as_in': vit_id}}

    optim_configs = {'inv_vit_bb': {'module_type': 'AdamW', 'lr': lr_bb},
                     'inv_vit_encoder': {'module_type': 'AdamW', 'lr': lr_enc},
                     'vision_transformer': {'module_type': 'AdamW', 'lr': lr_vit}}

    neptune_params = {'scope': get_parent_file(Path(__file__)), 'epochs': 0, 'batch_size': batch_size, 'seed': seed,
                      'dataset_id': 'imagenet1k', 'model_configs': model_configs, 'optim_configs': optim_configs,
                      'init_vit': 'vit_base_patch16_224', 'init_inv_bb': inv_bb_id,
                      'init_inv_enc': inv_enc_id, 'training_mode': training_mode,
                      'eval_loss_id': eval_loss_id, 'trade_off': trade_off}

    transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])
    transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()])

    dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
                                  num_workers=4, shuffle=True)

    dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=int(4 * args.batch_size), drop_last=False, pin_memory=True,
                                num_workers=4, shuffle=False)

    # dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train', transform=cu.vit_transforms)
    # dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='val', transform=cu.vit_transforms)
    # if run_id:
    #     checkpoint = nu.get_checkpoint(run_id=run_id)
    #     neptune_params = nu.get_params(run_id=run_id)
    #
    #     inv_bb, inv_bb_optim = nu.init_from_checkpoint(checkpoint, inv_bb_module, neptune_params)
    #     inv_enc, inv_enc_optim = nu.init_from_checkpoint(checkpoint, inv_enc_module, neptune_params)
    #     vit, vit_optim = nu.init_from_checkpoint(checkpoint, vit_module, neptune_params)
    #
    #     optims = [vit_optim, inv_bb_optim, inv_enc_optim]
    #     models = [vit, inv_bb, inv_enc]
    #
    #     best_loss = checkpoint['best_loss']
    #     test_step, train_step = checkpoint['test_step'], checkpoint['train_step']
    #     last_epoch = neptune_params['epochs']
    #     trade_off = neptune_params['trade_off']
    #
    #     training_mode = neptune_params['training_mode']
    #     # step_from = neptune_params['step_from']
    #     eval_loss_id = neptune_params['eval_loss_id']
    #     run = neptune.init_run(with_id=run_id, project=config.PROJECT, capture_hardware_metrics=False,
    #                            monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)
    # else:
    if inv_bb_id is not None:
        inv_bb = vu.init_model_from_neptune(run_id=inv_bb_id, module=inv_bb_module)
    else:
        inv_bb = nu.init_model_from_params(module=inv_bb_module, params=neptune_params)

    if inv_enc_id is not None:
        vit = vu.init_model_from_neptune(run_id=inv_enc_id, module=inv_enc_module)
    if vit_id is not None:
        vit = vu.init_model_from_neptune(run_id=vit_id, module=vit_module)
    vit, inv_bb, inv_enc = vu.init_vit_modules(vit_id=vit_id, inv_bb_id=inv_bb_id, inv_enc_id=inv_enc_id)
    vit.to(config.DEVICE), inv_bb.to(config.DEVICE), inv_enc.to(config.DEVICE)
    #
    vit = torch.compile(vit)
    inv_bb = torch.compile(inv_bb)
    inv_enc = torch.compile(inv_enc)

    vit_optim = nu.init_optim_from_params(vit, neptune_params)
    inv_bb_optim = nu.init_optim_from_params(inv_bb, neptune_params)
    inv_enc_optim = nu.init_optim_from_params(inv_enc, neptune_params)

    optims = [vit_optim, inv_bb_optim, inv_enc_optim]
    models = [vit, inv_bb, inv_enc]

    # nu.update_model_configs(params=neptune_params, modules=[inv_bb_module, inv_enc_module],
    #                         run_ids=[inv_bb_id, inv_enc_id])

    # run = neptune.init_run(project=config.PROJECT, source_files=[str(Path(__file__))],
    #                        capture_hardware_metrics=False,
    #                        monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False)

    # run['params'] = neptune_params
    val_step, train_step, last_epoch = 0, 0, 0

        # nu.upload_checkpoint(models=models, optims=optims, best_loss=best_loss, run=run,
        #                      test_step=test_step, train_step=train_step)

    if training_mode == TRAINING_MODE_BACKWARDS_ONLY:
        comb_params = list(inv_enc.parameters()) + list(inv_bb.parameters())
        comb_optim = create_optimizer_v2(model_or_params=comb_params, lr=lr, opt='lamb')
    # vit = torch.compile(vit)
    # inv_bb = torch.compile(inv_bb)
    # inv_enc = torch.compile(inv_enc)
    #
    #
    scaler = NativeScaler()
    with wandb.init(project=config.PROJECT, config=neptune_params, settings=wandb.Settings(_disable_stats=True)) as run:
        nu.prepare_run(run)
        best_loss = test_finetuned_vit(inv_enc=inv_enc, inv_bb=inv_bb, vit=vit, dataloader=dataloader_val,
                                       trade_off=trade_off, step=val_step)
        nu.save_model_state_tuple(models=models, run=run, model_id='ft_vit')

        for epoch in range(epochs):
            vit.train(), inv_enc.train(), inv_bb.train()
            for batch_id, (tensor, target) in enumerate(dataloader_train):
                tensor, target = tensor.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
                if training_mode == TRAINING_MODE_BACKWARDS_ONLY:
                    with torch.amp.autocast('cuda'):
                        with torch.no_grad():
                            enc_emb = vu.tensor_to_enc_emb(tensor, vit)

                        chain_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
                        chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=tensor)
                        comb_optim.zero_grad()
                        scaler(chain_recon_loss, comb_optim, clip_grad=1.0, parameters=comb_params)
                        wandb.log({f'train/recon_loss': chain_recon_loss}, step=train_step)
                else:
                    bb_emb = vu.tensor_to_bb_emb(tensor, vit)
                    enc_emb = vu.bb_emb_to_enc_emb(bb_emb, vit)
                    vit_out = vu.enc_emb_to_vit_out(enc_emb, vit)
                    class_loss = F.cross_entropy(input=vit_out, target=target)

                    chain_recon = vu.enc_emb_to_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb)
                    chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=tensor)
                    loss = (1 - trade_off) * class_loss + trade_off * chain_recon_loss

                    if trade_off == 0.0:
                        _zero_grad_optims()
                        chain_recon_loss.backward(retain_graph=True)
                        _clip_grads()
                        inv_enc_optim.step()
                        inv_bb_optim.step()
                        _zero_grad_optims()
                        class_loss.backward()
                        _clip_grads()
                        vit_optim.step()
                    else:
                        _zero_grad_optims()
                        loss.backward(retain_graph=True)
                        _clip_grads()
                        for optim in optims:
                            optim.step()
                    wandb.log({f'train/loss': loss}, step=train_step)
                    wandb.log({f'train/class_loss': class_loss}, step=train_step)
                    wandb.log({f'train/recon_loss': chain_recon_loss}, step=train_step)
                train_step += 1
            loss = test_finetuned_vit(inv_enc=inv_enc, inv_bb=inv_bb, vit=vit, dataloader=dataloader_val,
                                      trade_off=trade_off, step=val_step)
            val_step += 1
            if loss < best_loss:
                nu.save_model_state_tuple(models=models, run=run, model_id='ft_vit')

                # nu.upload_model_states(models=models, run=run)
                best_loss = loss
            # nu.upload_checkpoint(models=models, optims=optims, best_loss=best_loss, run=run, test_step=test_step,
            #                      train_step=train_step)
