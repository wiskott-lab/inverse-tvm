import config
import torch
from torch.functional import F

from modules.detr.util.misc import NestedTensor
from tools import detr_utils as du
from modules.detr.models.detr import PostProcess
from modules.detr.datasets import get_coco_api_from_dataset
from modules.detr.datasets.coco_eval import CocoEvaluator
import tools.coco_utils as cu


@torch.no_grad()
def test_finetuned_detr(detr, inv_bb, inv_enc, inv_dec, criterion, dataloader, run=None, test_loss='bb'):
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

    if run:
        run["test/detr_loss"].append(sum_detr_loss / num_inputs)
        run["test/class_loss"].append(sum_class_loss / num_inputs)
        run["test/giou_loss"].append(sum_giou_loss / num_inputs)
        run["test/bbox_loss"].append(sum_bbox_loss / num_inputs)
        run["test/bb_recon_loss"].append(sum_bb_recon_loss / num_inputs)
        run["test/enc_recon_loss"].append(sum_enc_recon_loss / num_inputs)
        run["test/dec_recon_loss"].append(sum_dec_recon_loss / num_inputs)
        run["test/chain_recon_loss"].append(sum_chain_recon_loss / num_inputs)

        coco_eval_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        for eval_string, eval_stat in zip(cu.coco_eval_strings, coco_eval_stats):
            run["test/" + eval_string].append(eval_stat)
    if test_loss == 'chain':
        return sum_chain_recon_loss / num_inputs
    elif test_loss == 'bb':
        return sum_bb_recon_loss / num_inputs
    else:
        raise NameError(f"Unknown eval_loss_id {test_loss}")


@torch.no_grad()
def test_finetuned_detr_recons(detr, inv_bb, inv_enc, inv_dec,  dataloader, run=None):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), detr.eval()
    sum_chain_recon_loss, num_inputs = 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        img = inputs.to(config.DEVICE)
        dec_emb, pos, mask = du.nested_tensor_to_dec_emb(img, detr)
        chain_recon = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc,
                                               inv_bb=inv_bb)
        chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=img.tensors)
        sum_chain_recon_loss += chain_recon_loss * len(inputs.tensors)
        num_inputs += len(inputs.tensors)

    if run:

        run["test/chain_recon_loss"].append(sum_chain_recon_loss / num_inputs)
    return sum_chain_recon_loss / num_inputs


@torch.no_grad()
def test_finetuned_detr_recons_imgnet(detr, inv_bb, inv_enc, inv_dec,  dataloader, run=None):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), detr.eval()
    sum_chain_recon_loss, num_inputs = 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        mask = torch.zeros(size=(inputs.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.bool)
        img = NestedTensor(inputs, mask).to(config.DEVICE)
        dec_emb, pos, mask = du.nested_tensor_to_dec_emb(img, detr)
        chain_recon = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_dec=inv_dec, mask=mask, pos=pos, inv_enc=inv_enc, inv_bb=inv_bb)
        chain_recon_loss = F.mse_loss(input=cu.normalize(chain_recon), target=img.tensors)
        sum_chain_recon_loss += chain_recon_loss * len(img.tensors)
        num_inputs += len(img.tensors)
    if run:
        run["test/chain_recon_loss"].append(sum_chain_recon_loss / num_inputs)
    return sum_chain_recon_loss / num_inputs


@torch.no_grad()
def test_train_in_parallel(detr, inv_bb, inv_enc, inv_dec, inv_detect, dataloader, run=None):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), inv_detect.eval(), detr.eval()
    sum_bb_loss, sum_enc_loss, sum_dec_loss, sum_detect_loss, num_inputs = 0, 0, 0, 0, 0
    # sum_chain_bb_loss, sum_chain_enc_loss, sum_chain_dec_loss, sum_chain_detect_loss = 0, 0, 0, 0
    # sum_chain_bb_loss_denorm, sum_chain_enc_loss_denorm, sum_chain_dec_loss_denorm, sum_chain_detect_loss_denorm = 0, 0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        nested_tensor = inputs.to(config.DEVICE)
        bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
        enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, mask=mask, pos=pos, detr=detr)
        dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
        detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
        nested_tensor_recon = du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=inv_bb)
        bb_emb_recon = du.enc_emb_to_bb_emb(enc_emb=enc_emb, inv_enc=inv_enc, pos=pos, mask=mask)
        enc_emb_recon = du.dec_emb_to_enc_emb(dec_emb=dec_emb, inv_dec=inv_dec, pos=pos)
        dec_emb_recon = du.detr_out_to_dec_emb(detr_out=detr_out, inv_detect=inv_detect)
        bb_loss = F.mse_loss(input=cu.normalize(nested_tensor_recon), target=nested_tensor.tensors)
        enc_loss = F.mse_loss(input=bb_emb_recon, target=bb_emb)
        dec_loss = F.mse_loss(input=enc_emb_recon, target=enc_emb)
        detect_loss = F.mse_loss(input=dec_emb_recon[-1], target=dec_emb[-1])
        num_inputs += len(inputs.tensors)
        sum_bb_loss += bb_loss * len(inputs.tensors)
        sum_enc_loss += enc_loss * len(inputs.tensors)
        sum_dec_loss += dec_loss * len(inputs.tensors)
        sum_detect_loss += detect_loss * len(inputs.tensors)

        # chain_bb_emb = du.bb_emb_to_img_tensor(bb_emb=bb_emb,  inv_bb=inv_bb)
        # chain_enc_emb = du.enc_emb_to_img_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos)
        # chain_dec_emb = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos, inv_dec=inv_dec)
        # chain_detect_emb = du.detr_out_to_img_tensor(detr_out=detr_out, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos, inv_dec=inv_dec, inv_detect=inv_detect)
        #
        # bb_loss = F.mse_loss(input=cu.normalize(chain_bb_emb), target=nested_tensor.tensors)
        # bb_loss_denorm = F.mse_loss(input=chain_bb_emb, target=cu.denormalize(nested_tensor.tensors))
        # sum_chain_bb_loss += bb_loss * len(inputs.tensors)
        # sum_chain_bb_loss_denorm += bb_loss_denorm * len(inputs.tensors)
        #
        # enc_loss = F.mse_loss(input=cu.normalize(chain_enc_emb), target=nested_tensor.tensors)
        # enc_loss_denorm = F.mse_loss(input=chain_enc_emb, target=cu.denormalize(nested_tensor.tensors))
        # sum_chain_enc_loss += enc_loss * len(inputs.tensors)
        # sum_chain_enc_loss_denorm += enc_loss_denorm * len(inputs.tensors)
        #
        # dec_loss = F.mse_loss(input=cu.normalize(chain_dec_emb), target=nested_tensor.tensors)
        # dec_loss_denorm = F.mse_loss(input=chain_dec_emb, target=cu.denormalize(nested_tensor.tensors))
        # sum_chain_dec_loss += dec_loss * len(inputs.tensors)
        # sum_chain_dec_loss_denorm += dec_loss_denorm * len(inputs.tensors)
        #
        # detect_loss = F.mse_loss(input=cu.normalize(chain_detect_emb), target=nested_tensor.tensors)
        # detect_loss_denorm = F.mse_loss(input=chain_detect_emb, target=cu.denormalize(nested_tensor.tensors))
        # sum_chain_detect_loss += detect_loss * len(inputs.tensors)
        # sum_chain_detect_loss_denorm += detect_loss_denorm * len(inputs.tensors)

    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/enc_loss"].append((sum_enc_loss / num_inputs).item())
        run["test/dec_loss"].append((sum_dec_loss / num_inputs).item())
        run["test/detect_loss"].append((sum_detect_loss / num_inputs).item())

        # run["test/bb_loss_chain"].append((sum_chain_bb_loss / num_inputs).item())
        # run["test/enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())
        # run["test/dec_loss_chain"].append((sum_chain_dec_loss / num_inputs).item())
        # run["test/detect_loss_chain"].append((sum_chain_detect_loss / num_inputs).item())
        #
        # run["test/bb_loss_chain_denorm"].append((sum_chain_bb_loss_denorm / num_inputs).item())
        # run["test/enc_loss_chain_denorm"].append((sum_chain_enc_loss_denorm / num_inputs).item())
        # run["test/dec_loss_chain_denorm"].append((sum_chain_dec_loss_denorm / num_inputs).item())
        # run["test/detect_loss_chain_denorm"].append((sum_chain_detect_loss_denorm / num_inputs).item())
    return (sum_bb_loss / num_inputs).item(), (sum_enc_loss / num_inputs).item(), (sum_dec_loss / num_inputs).item(), (sum_detect_loss / num_inputs).item()


@torch.no_grad()
def test_train_chain_losses(detr, inv_bb, inv_enc, inv_dec, inv_detect, dataloader, run=None):
    inv_bb.eval(), inv_enc.eval(), inv_dec.eval(), inv_detect.eval(), detr.eval()
    num_inputs = 0
    sum_chain_bb_loss, sum_chain_enc_loss, sum_chain_dec_loss, sum_chain_detect_loss = 0, 0, 0, 0
    sum_chain_bb_loss_denorm, sum_chain_enc_loss_denorm, sum_chain_dec_loss_denorm, sum_chain_detect_loss_denorm = 0, 0, 0, 0

    for batch_id, (inputs, targets) in enumerate(dataloader):
        nested_tensor = inputs.to(config.DEVICE)
        bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
        enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, mask=mask, pos=pos, detr=detr)
        dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
        detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)

        chain_bb_emb = du.bb_emb_to_img_tensor(bb_emb=bb_emb,  inv_bb=inv_bb)
        chain_enc_emb = du.enc_emb_to_img_tensor(enc_emb=enc_emb, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos)
        chain_dec_emb = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos, inv_dec=inv_dec)
        chain_detect_emb = du.detr_out_to_img_tensor(detr_out=detr_out, inv_enc=inv_enc, inv_bb=inv_bb, mask=mask, pos=pos, inv_dec=inv_dec, inv_detect=inv_detect)

        bb_loss = F.mse_loss(input=cu.normalize(chain_bb_emb), target=nested_tensor.tensors)
        bb_loss_denorm = F.mse_loss(input=chain_bb_emb, target=cu.denormalize(nested_tensor.tensors))
        sum_chain_bb_loss += bb_loss * len(inputs.tensors)
        sum_chain_bb_loss_denorm += bb_loss_denorm * len(inputs.tensors)

        enc_loss = F.mse_loss(input=cu.normalize(chain_enc_emb), target=nested_tensor.tensors)
        enc_loss_denorm = F.mse_loss(input=chain_enc_emb, target=cu.denormalize(nested_tensor.tensors))
        sum_chain_enc_loss += enc_loss * len(inputs.tensors)
        sum_chain_enc_loss_denorm += enc_loss_denorm * len(inputs.tensors)

        dec_loss = F.mse_loss(input=cu.normalize(chain_dec_emb), target=nested_tensor.tensors)
        dec_loss_denorm = F.mse_loss(input=chain_dec_emb, target=cu.denormalize(nested_tensor.tensors))
        sum_chain_dec_loss += dec_loss * len(inputs.tensors)
        sum_chain_dec_loss_denorm += dec_loss_denorm * len(inputs.tensors)

        detect_loss = F.mse_loss(input=cu.normalize(chain_detect_emb), target=nested_tensor.tensors)
        detect_loss_denorm = F.mse_loss(input=chain_detect_emb, target=cu.denormalize(nested_tensor.tensors))
        sum_chain_detect_loss += detect_loss * len(inputs.tensors)
        sum_chain_detect_loss_denorm += detect_loss_denorm * len(inputs.tensors)

        num_inputs += len(inputs.tensors)

    if run:
        run["test/bb_loss_chain"].append((sum_chain_bb_loss / num_inputs).item())
        run["test/enc_loss_chain"].append((sum_chain_enc_loss / num_inputs).item())
        run["test/dec_loss_chain"].append((sum_chain_dec_loss / num_inputs).item())
        run["test/detect_loss_chain"].append((sum_chain_detect_loss / num_inputs).item())

        run["test/bb_loss_chain_denorm"].append((sum_chain_bb_loss_denorm / num_inputs).item())
        run["test/enc_loss_chain_denorm"].append((sum_chain_enc_loss_denorm / num_inputs).item())
        run["test/dec_loss_chain_denorm"].append((sum_chain_dec_loss_denorm / num_inputs).item())
        run["test/detect_loss_chain_denorm"].append((sum_chain_detect_loss_denorm / num_inputs).item())


@torch.no_grad()
def test_classic_in_par(detr, bb_to_img, enc_to_img, dec_to_img, detr_out_to_img, dataloader, run=None):
    sum_bb_loss, sum_enc_loss, sum_dec_loss, sum_detect_loss,  sum_bb_loss_denorm, sum_enc_loss_denorm, sum_dec_loss_denorm, sum_detect_loss_denorm, num_inputs = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        nested_tensor = inputs.to(config.DEVICE)
        bb_emb, pos, mask = du.nested_tensor_to_bb_emb(nested_tensor=nested_tensor, detr=detr)
        enc_emb = du.bb_emb_to_enc_emb(bb_emb=bb_emb, mask=mask, pos=pos, detr=detr)
        dec_emb = du.enc_emb_to_dec_emb(enc_emb=enc_emb, detr=detr, mask=mask, pos=pos)
        detr_out = du.dec_emb_to_detr_out(dec_emb=dec_emb, detr=detr)
        if bb_to_img is not None:
            bb_to_img_recon = du.bb_emb_to_img_tensor(bb_emb=bb_emb, inv_bb=bb_to_img[0])
            bb_loss = F.mse_loss(input=cu.normalize(bb_to_img_recon), target=nested_tensor.tensors)
            sum_bb_loss += bb_loss * len(inputs.tensors)
            bb_loss_denorm = F.mse_loss(input=bb_to_img_recon, target=cu.denormalize(nested_tensor.tensors))
            sum_bb_loss_denorm += bb_loss_denorm * len(inputs.tensors)
        if enc_to_img is not None:
            enc_to_img_recon = du.enc_emb_to_img_tensor(enc_emb=enc_emb, inv_bb=enc_to_img[0], inv_enc=enc_to_img[1],
                                                        mask=mask, pos=pos)
            enc_loss = F.mse_loss(input=cu.normalize(enc_to_img_recon), target=nested_tensor.tensors)
            sum_enc_loss += enc_loss * len(inputs.tensors)
            enc_loss_denorm = F.mse_loss(input=enc_to_img_recon, target=cu.denormalize(nested_tensor.tensors))
            sum_enc_loss_denorm += enc_loss_denorm * len(inputs.tensors)
        if dec_to_img is not None:
            dec_to_img_recon = du.dec_emb_to_img_tensor(dec_emb=dec_emb, inv_bb=dec_to_img[0], inv_enc=dec_to_img[1],
                                                        inv_dec=dec_to_img[2], mask=mask, pos=pos)
            dec_loss = F.mse_loss(input=cu.normalize(dec_to_img_recon), target=nested_tensor.tensors)
            sum_dec_loss += dec_loss * len(inputs.tensors)
            dec_loss_denorm = F.mse_loss(input=dec_to_img_recon, target=cu.denormalize(nested_tensor.tensors))
            sum_dec_loss_denorm += dec_loss_denorm * len(inputs.tensors)
        if detr_out_to_img is not None:
            detr_out_to_img_recon = du.detr_out_to_img_tensor(detr_out=detr_out, inv_bb=detr_out_to_img[0],
                                                              inv_enc=detr_out_to_img[1], inv_dec=detr_out_to_img[2],
                                                              inv_detect=detr_out_to_img[3], mask=mask, pos=pos)
            detect_loss = F.mse_loss(input=cu.normalize(detr_out_to_img_recon), target=nested_tensor.tensors)
            sum_detect_loss += detect_loss * len(inputs.tensors)
            detect_loss_denorm = F.mse_loss(input=detr_out_to_img_recon, target=cu.denormalize(nested_tensor.tensors))
            sum_detect_loss_denorm += detect_loss_denorm * len(inputs.tensors)
        num_inputs += len(inputs.tensors)
    if run:
        run["test/bb_loss"].append((sum_bb_loss / num_inputs).item())
        run["test/enc_loss"].append((sum_enc_loss / num_inputs).item())
        run["test/dec_loss"].append((sum_dec_loss / num_inputs).item())
        run["test/detect_loss"].append((sum_detect_loss / num_inputs).item())
        run["test/bb_denorm_loss"].append((sum_bb_loss_denorm / num_inputs).item())
        run["test/enc_denorm_loss"].append((sum_enc_loss_denorm / num_inputs).item())
        run["test/dec_denorm_loss"].append((sum_dec_loss_denorm / num_inputs).item())
        run["test/detect_denorm_loss"].append((sum_detect_loss_denorm / num_inputs).item())
    return (sum_bb_loss / num_inputs).item(), (sum_enc_loss / num_inputs).item(), (sum_dec_loss / num_inputs).item(), (sum_detect_loss / num_inputs).item()