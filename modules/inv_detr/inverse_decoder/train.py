import torch
import config
from detr.hubconf import detr_resnet50
import tools.coco_utils as cu
import argparse
from torch.functional import F
from inverse_decoder.utils import test_inv_dec
from inverse_decoder import models as inv_dec_module
from tools import training_utils, detr_utils
from tools import detr_utils as du
from torch.optim import Adam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=0.0001)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=128)
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    cuda_device = args.cuda_device
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
    seed = training_utils.init_seeds(None)
    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_val = cu.get_dataloader(batch_size=batch_size, dataset_type='val')

    detr = detr_resnet50(pretrained=True)
    detr.to(config.DEVICE).eval()

    inv_dec = inv_dec_module.InverseTransformerDecoder()
    inv_dec_optim = Adam(params=inv_dec.parameters(), lr=lr)

    inv_dec.to(config.DEVICE)
    best_loss = test_inv_dec(inv_dec=inv_dec,  detr=detr, dataloader=dataloader_val)
    test_step, train_step, last_epoch = 0, -1, 0
    for epoch in range(epochs):
        detr.train(), inv_dec.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                enc_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor, detr)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, pos=pos, mask=mask)
            recon = detr_utils.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
            loss = F.mse_loss(input=recon, target=enc_emb)
            training_utils.optim_step(inv_dec_optim, loss)
            train_step += 1
        loss = test_inv_dec(inv_dec=inv_dec,  detr=detr, dataloader=dataloader_val)
        test_step += 1
        if loss < best_loss:
            best_loss = loss
