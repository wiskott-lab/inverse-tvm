import torch
import config
import tools.training_utils as tu
import tools.coco_utils as cu
import argparse
from torch.functional import F
from modules.inv_vit_encoder.utils import test_inv_enc
from modules.inv_vit_encoder import models as inv_vit_enc_module
import tools.vit_utils as vu
import timm
from torch.optim import Adam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = tu.init_seeds(None)

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train', transform=cu.vit_transforms)
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='val', transform=cu.vit_transforms)

    vit = timm.create_model("vit_base_patch16_224" , pretrained=True)
    vit.to(config.DEVICE)

    inv_enc = inv_vit_enc_module.InverseViTEncoder()
    inv_enc_optim = Adam(params=inv_enc.parameters(), lr=lr)
    best_loss = test_inv_enc(inv_enc=inv_enc, vit=vit, dataloader=dataloader_test)
    inv_enc.to(config.DEVICE)
    test_step, train_step, last_epoch = 0, -1, 0
    for epoch in range(epochs):
        vit.train(), inv_enc.train()
        for batch_id, (tensor, _) in enumerate(dataloader_train):
            tensor = tensor.tensors.to(config.DEVICE)
            with torch.no_grad():
                input_emb = vu.tensor_to_bb_emb(tensor, vit)
                enc_emb = vu.tensor_to_enc_emb(tensor, vit)
            recon = vu.enc_emb_to_bb_emb(enc_emb, inv_enc)
            loss = F.mse_loss(input=recon, target=input_emb)
            tu.optim_step(inv_enc_optim, loss)
            train_step += 1
        loss = test_inv_enc(inv_enc=inv_enc, vit=vit, dataloader=dataloader_test)
        test_step += 1
        if loss < best_loss:
            best_loss = loss