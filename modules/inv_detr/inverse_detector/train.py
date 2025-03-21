import torch
import config
import tools.training_utils as training_utils
from modules.detr.datasets.coco import CocoDetection as DetrCocoDetection
from modules.detr.datasets.coco import make_coco_transforms
import argparse
from torch.functional import F
from pathlib import Path
import neptune
import tools.neptune_utils as nu
from modules.inverse_detector.utils import eval_inverse_detector
from modules.inverse_detector import models as inverse_detector_module
import tools.detr_utils as du
import tools.coco_utils as cu
from tools.misc_utils import get_parent_file
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
    train_detr_in_eval_mode = args.train_detr_in_eval_mode
    train_inverse_detector_in_eval_mode = args.train_inverse_detector_in_eval_mode

    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)

    seed = training_utils.init_seeds(None)

    transform_train = make_coco_transforms('inv_backbone_train')
    transform_val = make_coco_transforms('inv_backbone_val')

    dataloader_train = cu.get_dataloader(dataset_type='train', batch_size=batch_size)
    dataloader_test = cu.get_dataloader(dataset_type='test', batch_size=batch_size)


    detr = detr_resnet50(pretrained=True)
    detr.to(config.DEVICE)
    inv_detc = inverse_detector_module.InverseDetector()
    inverse_detector_optim = Adam(lr=lr, params=inv_detc.parameters())
    best_loss = eval_inverse_detector(model=inv_detc, detr=detr, dataloader=dataloader_test)
    inv_detc.to(config.DEVICE)
    test_step, train_step, last_epoch = 0, -1, 0
    for epoch in range(epochs):
        inv_detc.train()
        for batch_id, (inputs, _) in enumerate(dataloader_train):
            nested_tensor = inputs.to(config.DEVICE)
            with torch.no_grad():
                dec_emb = du.nested_tensor_to_dec_emb(nested_tensor=nested_tensor, detr=detr)
                detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
            recon = inv_detc(detr_out["pred_logits"], detr_out["pred_boxes"])
            loss = F.mse_loss(input=recon, target=dec_emb[-1].transpose(0, 1))
            training_utils.optim_step(inverse_detector_optim, loss)
            train_step += 1
        loss = eval_inverse_detector(model=inv_detc,  detr=detr, dataloader=dataloader_test)
        test_step += 1
        if loss < best_loss:
            best_loss = loss