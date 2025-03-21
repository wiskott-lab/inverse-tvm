import config
import torch
from torch.functional import F
from tools import detr_utils as du
from modules.detr.models.detr import PostProcess
from modules.detr.datasets import get_coco_api_from_dataset
from modules.detr.datasets.coco_eval import CocoEvaluator
import tools.coco_utils as cu


@torch.no_grad()
def test_finetuned_detr(detr, inv_bb, inv_enc, inv_dec, criterion, dataloader, test_loss='bb'):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), detr.eval(), criterion.eval()
    base_ds = get_coco_api_from_dataset(dataloader.dataset)
    coco_evaluator = CocoEvaluator(coco_gt=base_ds, iou_types=('bbox',))
    postprocessors = {'bbox': PostProcess()}
    sum_class_loss, sum_bbox_loss, sum_giou_loss, sum_detr_loss, sum_enc_recon_loss, sum_bb_recon_loss,\
        sum_dec_recon_loss, sum_chain_recon_loss, num_inputs = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
        bb_emb, pos, mask = du.nested_tensor_to_bb_emb(img, detr)
        enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, detr=detr, mask=mask, pos=pos)
        dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
        detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)

        enc_emb_recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, pos=pos, inv_dec=inv_dec)
        bb_emb_recon = du.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, mask=mask, pos=pos)
        img_recon = du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb)

        chain_recon = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc,
                                               inv_bb=inv_bb)

        enc_emb_recon_loss = F.mse_loss(input=enc_emb_recon, target=enc_emb)
        bb_emb_recon_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
        img_recon_loss = F.mse_loss(input=img_recon, target=img.tensors)
        chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=img.tensors)

        loss_dict = criterion(detr_out, targets)
        weight_dict = criterion.weight_dict
        detr_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        sum_detr_loss += detr_loss * len(inputs.tensors)
        sum_class_loss += loss_dict['loss_ce'] * len(inputs.tensors)
        sum_bbox_loss += loss_dict['loss_bbox'] * len(inputs.tensors)
        sum_giou_loss += loss_dict['loss_giou'] * len(inputs.tensors)
        sum_enc_recon_loss += bb_emb_recon_loss * len(inputs.tensors)
        sum_dec_recon_loss += enc_emb_recon_loss * len(inputs.tensors)
        sum_chain_recon_loss += chain_recon_loss * len(inputs.tensors)
        sum_bb_recon_loss += img_recon_loss * len(inputs.tensors)
        num_inputs += len(inputs.tensors)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](detr_out, orig_target_sizes.to(
            config.DEVICE))  # removes the no-class token and then takes the largest score, i.e., if no class
        # was predicted for a certain bounding box, another class with very low confidence will be used instead
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)
        # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    if test_loss == 'chain':
        return sum_chain_recon_loss / num_inputs
    elif test_loss == 'bb':
        return sum_bb_recon_loss / num_inputs
    else:
        raise NameError(f"Unknown eval_loss_id {test_loss}")
