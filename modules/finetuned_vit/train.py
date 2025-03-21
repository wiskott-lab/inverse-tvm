import torch
import config
import tools.training_utils as tu
import tools.coco_utils as cu
import argparse
from torch.functional import F
from modules.finetuned_vit.utils import test_finetuned_vit
from modules.inv_vit_bb import models as inv_bb_module
from modules.inv_vit_encoder import models as inv_enc_module
import timm
from torch.utils.data import DataLoader
import tools.vit_utils as vu
from torchvision import transforms, datasets
from torch.optim import Adam

TRAINING_MODE_CHAIN = 'chain'
TRAINING_MODE_SUM = 'sum'
TRAINING_MODE_ISOLATED = 'isolated'
TRAINING_MODE_BACKWARDS_ONLY = 'backwards_only'


def _zero_grad_optims():
    for optim in optims:
        optim.zero_grad()


def _clip_grads():
    for model in models:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=64)

    parser.add_argument("--learning_rate_bb", "-lr-back", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_enc", "-lr-enc", help='learning rate', type=float, default=0.00001)
    parser.add_argument("--learning_rate_vit", "-lr-custom_vit", help='learning rate', type=float, default=0.00001)

    parser.add_argument("--trade_off", "-t", help='trade-off', type=float, default=0.9)

    parser.add_argument("--training_mode", "-tm", help='training mode', type=str, default=TRAINING_MODE_CHAIN)
    parser.add_argument("--eval_loss_id", "-eid", help='eval_loss_id', type=str, default='enc_emb')

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    lr_bb = args.learning_rate_bb
    lr_enc = args.learning_rate_enc
    lr_vit = args.learning_rate_vit

    trade_off = args.trade_off

    training_mode = args.training_mode
    # step_from = args.step_from
    eval_loss_id = args.eval_loss_id

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = tu.init_seeds(None)

    transform_train = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    dataset_train = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_TRAIN_SPLIT, transform=transform_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,
                                  num_workers=4, shuffle=True)


    dataset_val = datasets.ImageNet(config.IMGNET1k_DIR, split=config.IMGNET1k_VAL_SPLIT, transform=transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=int(4 * args.batch_size), drop_last=False, pin_memory=True,
                                num_workers=4, shuffle=False)

    vit = timm.create_model("vit_base_patch16_224" , pretrained=True)
    inv_bb = inv_bb_module.InverseViTBackbone()
    inv_enc = inv_enc_module.InverseViTEncoder()
    vit.to(config.DEVICE), inv_bb.to(config.DEVICE), inv_enc.to(config.DEVICE)

    vit_optim = Adam(params=vit.parameters(), lr=lr_vit)
    inv_bb_optim = Adam(params=inv_bb.parameters(), lr=lr_bb)
    inv_enc_optim = Adam(params=inv_enc.parameters(), lr=lr_enc)

    optims = [vit_optim, inv_bb_optim, inv_enc_optim]
    models = [vit, inv_bb, inv_enc]


    best_loss = test_finetuned_vit(inv_enc=inv_enc, inv_bb=inv_bb, vit=vit, dataloader=dataloader_val, trade_off=trade_off)
    test_step, train_step, last_epoch = 0, -1, 0

    for epoch in range(epochs):
        vit.train(), inv_enc.train(), inv_bb.train()
        for batch_id, (tensor, target) in enumerate(dataloader_train):
            tensor, target = tensor.to(config.DEVICE), target.to(config.DEVICE)
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
            train_step += 1
        loss = test_finetuned_vit(inv_enc=inv_enc, inv_bb=inv_bb, vit=vit, dataloader=dataloader_val, trade_off=trade_off)
        test_step += 1
        if loss < best_loss:
            best_loss = loss
