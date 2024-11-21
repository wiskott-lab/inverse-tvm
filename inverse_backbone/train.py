import torch
from inverse_backbone import models as inverse_backbone_module
from inverse_backbone.utils import test_inv_bb
from detr.hubconf import detr_resnet50
import argparse
from torch.functional import F
from utils import detr_utils as du, coco_utils as cu, training_utils
from torch.optim import Adam
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--path", "-p", help='path for storing trained model', type=str)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=3e-3)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=32)

    args = parser.parse_args()
    epochs = args.epochs
    path = args.path
    lr = args.learning_rate
    batch_size = args.batch_size
    device = args.device

    seed = training_utils.init_seeds(None)
    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')

    detr = detr_resnet50(pretrained=True)
    inv_bb = inverse_backbone_module.EnhancedBatchNormalizedConvolutionalDecoder()
    torch.save(deepcopy(inv_bb.state_dict()), path)
    inv_bb_optim = Adam(params=inv_bb.parameters(), lr=lr)
    detr.to(device), inv_bb.to(device)
    best_loss = test_inv_bb(inv_bb=inv_bb, detr=detr, dataloader=dataloader_test, device=device)
    for epoch in range(epochs):
        inv_bb.train(), detr.train()
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(device)
            with torch.no_grad():
                bb_emb, _, _ = du.nested_tensor_to_bb_emb(nested_tensor, detr)
            recon = du.normalize(du.bb_emb_to_img(bb_emb=bb_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=nested_tensor.tensors)
            training_utils.optim_step(inv_bb_optim, loss)
        loss = test_inv_bb(inv_bb=inv_bb, detr=detr, dataloader=dataloader_test, device=device)
        if loss < best_loss:
            torch.save(deepcopy(inv_bb.state_dict()), path)
            best_loss = loss
