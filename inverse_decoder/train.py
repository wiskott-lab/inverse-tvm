import torch
from detr.hubconf import detr_resnet50
import argparse
from torch.functional import F
from inverse_decoder.utils import test_inv_dec
from inverse_decoder import models as inv_dec_module
from utils import detr_utils as du, coco_utils as cu, training_utils
from torch.optim import Adam
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", help='device', type=str, default='cpu')
    parser.add_argument("--path", "-p", help='path for storing trained model', type=str)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.learning_rate
    path = args.path
    batch_size = args.batch_size
    device = args.device

    seed = training_utils.init_seeds(None)

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')

    detr = detr_resnet50(pretrained=True)
    inv_dec = inv_dec_module.InverseTransformerDecoder()
    torch.save(deepcopy(inv_dec.state_dict()), path)
    inv_dec_optim = Adam(params=inv_dec.parameters(), lr=lr)
    inv_dec.to(device), detr.to(device)
    best_loss = test_inv_dec(inv_dec=inv_dec, detr=detr, dataloader=dataloader_test, device=device)
    for epoch in range(epochs):
        inv_dec.train(), detr.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(device)
            with torch.no_grad():
                enc_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, pos=pos, mask=mask)
            recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
            loss = F.mse_loss(input=recon, target=enc_emb)
            training_utils.optim_step(inv_dec_optim, loss)
        loss = test_inv_dec(inv_dec=inv_dec, detr=detr, dataloader=dataloader_test, device=device)
        if loss < best_loss:
            torch.save(deepcopy(inv_dec.state_dict()), path)
            best_loss = loss
