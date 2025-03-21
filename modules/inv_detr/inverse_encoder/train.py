import torch
import config
import tools.training_utils as tu
import tools.coco_utils as cu
import argparse
from torch.functional import F
from modules.inverse_encoder.utils import test_inv_enc
from modules.inverse_encoder import models as inverse_encoder_module
import tools.detr_utils as du
from modules.detr.hubconf import detr_resnet50
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

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_val = cu.get_dataloader(batch_size=batch_size, dataset_type='val')

    detr = detr_resnet50(pretrained=True)
    detr.to(config.DEVICE)
    inv_enc = inverse_encoder_module.InverseTransformerEncoder()
    inv_enc_optim = Adam(lr=lr, params=inv_enc.parameters())
    best_loss = test_inv_enc(inv_enc=inv_enc, detr=detr, dataloader=dataloader_val)
    inv_enc.to(config.DEVICE)
    test_step, train_step, last_epoch = 0, -1, 0
    for epoch in range(epochs):
        detr.train(), inv_enc.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
                enc_emb = du.bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
            recon = inv_enc(enc_emb, src_key_padding_mask=mask, pos=pos)
            loss = F.mse_loss(input=recon, target=bb_emb)
            tu.optim_step(inv_enc_optim, loss)
            train_step += 1
        loss = test_inv_enc(inv_enc=inv_enc, detr=detr, dataloader=dataloader_val)
        test_step += 1
        if loss < best_loss:
            best_loss = loss
