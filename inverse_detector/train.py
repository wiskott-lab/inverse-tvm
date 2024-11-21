import torch
import argparse
from torch.functional import F
from inverse_detector.utils import test_inverse_detector
from inverse_detector import models as inverse_detector_module
from utils import detr_utils as du, coco_utils as cu, training_utils as training_utils
from detr.hubconf import detr_resnet50
from torch.optim import Adam
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", help='device', type=str, default='cpu')
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
    inverse_detector = inverse_detector_module.InverseDetector()
    torch.save(deepcopy(inverse_detector.state_dict()), path)
    inverse_detector_optim = Adam(lr=lr, params=inverse_detector.parameters())
    detr.to(device), inverse_detector.to(device)
    best_loss = test_inverse_detector(model=inverse_detector, detr=detr, dataloader=dataloader_test, device=device)
    for epoch in epochs:
        inverse_detector.train(), detr.train()
        for batch_id, (inputs, _) in enumerate(dataloader_train):
            x = inputs.to(device)
            with torch.no_grad():
                bb_emb, pos, mask = du.nested_tensor_to_bb_emb(x, detr)
                enc_emb = du.bb_emb_to_enc_emb(bb_emb, detr, mask, pos)
                dec_emb = du.enc_emb_to_dec_emb(enc_emb, detr, mask, pos)
                detr_out = du.dec_emb_to_detr_out(dec_emb, detr)
            recon = inverse_detector(detr_out["pred_logits"], detr_out["pred_boxes"])
            loss = F.mse_loss(input=recon, target=dec_emb[-1].transpose(0, 1))
            training_utils.optim_step(inverse_detector_optim, loss)
        loss = test_inverse_detector(model=inverse_detector, detr=detr, dataloader=dataloader_test, device=device)
        if loss < best_loss:
            best_loss = loss
            torch.save(deepcopy(inverse_detector.state_dict()), path)
