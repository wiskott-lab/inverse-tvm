import torch
import config
from modules.inverse_backbone import models as inverse_backbone_module
from modules.inverse_backbone.utils import test_inv_bb
from modules.detr.hubconf import detr_resnet50
import argparse
from torch.functional import F
from pathlib import Path
import uuid
from tools import detr_utils as du, training_utils as tu, coco_utils as cu
from torch.optim import Adam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--learning_rate", "-lr", help='learning rate', type=float, default=3e-3)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=100)
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=32)
    parser.add_argument("--sched", type=str, default="linearW")

    args = parser.parse_args()

    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    cuda_device = args.cuda_device

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    seed = tu.init_seeds(None)
    model_path = config.TMP_DIR / (str(uuid.uuid4()))  # tmp path for storing model state dict
    best_loss = 2147483647

    dataloader_train = cu.get_dataloader(batch_size=batch_size, dataset_type='train')
    dataloader_test = cu.get_dataloader(batch_size=batch_size, dataset_type='test')
    source_files = [str(Path(__file__))]

    detr = detr_resnet50(pretrained=True)
    backbone = detr.backbone
    input_proj = detr.input_proj
    backbone.to(config.DEVICE)
    input_proj.to(config.DEVICE)

    detr = detr_resnet50(pretrained=True)
    inv_bb = inverse_backbone_module.EnhancedBatchNormalizedConvolutionalDecoder()
    inv_bb_optim = Adam(params=inv_bb.parameters(), lr=lr)
    inv_bb.to(config.DEVICE)
    best_loss = test_inv_bb(inv_bb=inv_bb, detr=detr, dataloader=dataloader_test)
    test_step, train_step, last_epoch = 0, -1, 0
    for epoch in range(epochs):
        for batch_id, (nested_tensor, _) in enumerate(dataloader_train):
            nested_tensor = nested_tensor.to(config.DEVICE)
            with torch.no_grad():
                bb_emb, _, _ = du.nested_tensor_to_bb_emb(nested_tensor, detr)
            recon = cu.normalize(du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb))
            loss = F.mse_loss(input=recon, target=nested_tensor.tensors)  # may need to account for resizing here
            tu.optim_step(inv_bb_optim, loss)
            train_step += 1
        loss = test_inv_bb(inv_bb=inv_bb,detr=detr, dataloader=dataloader_test)
        test_step += 1
        if loss < best_loss:
            best_loss = loss