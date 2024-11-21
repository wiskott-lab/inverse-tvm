import torch
import argparse
from torch.functional import F
from inverse_encoder.utils import test_inv_enc
from inverse_encoder import models as inverse_encoder_module
from utils import detr_utils as du, coco_utils as cu, training_utils as tu
from detr.hubconf import detr_resnet50
from copy import deepcopy
from torch.optim import Adam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", help='device', type=str, default='cpu')
    parser.add_argument("--path", "-p", help='path for storing trained model', type=str)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.001)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)

    args = parser.parse_args()
    epochs = args.epochs
    path = args.path
    lr = args.learning_rate
    batch_size = args.batch_size
    device = args.device

    seed = tu.init_seeds(None)
    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')
    detr = detr_resnet50(pretrained=True)
    inv_enc = inverse_encoder_module.InverseTransformerEncoder()
    torch.save(deepcopy(inv_enc.state_dict()), path)
    inv_enc_optim = Adam(lr=lr, params=inv_enc.parameters())
    detr.to(device), inv_enc.to(device)

    best_loss = test_inv_enc(inv_enc=inv_enc, detr=detr,  dataloader=dataloader_test, device=device)

    for epoch in range(epochs):
        detr.train(), inv_enc.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(device)
            with torch.no_grad():
                bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
                enc_emb = du.bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
            recon = inv_enc(enc_emb, src_key_padding_mask=mask, pos=pos)
            loss = F.mse_loss(input=recon, target=bb_emb)
            tu.optim_step(inv_enc_optim, loss)
        loss = test_inv_enc(inv_enc=inv_enc, detr=detr, dataloader=dataloader_test, device=device)
        if loss < best_loss:
            torch.save(deepcopy(inv_enc.state_dict()), path)
            best_loss = loss
