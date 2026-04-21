import config
import torch
from torch.functional import F
import tools.detr_utils as detr_tools

def eval_inverse_detector(model, detr, dataloader, run=None):
    model.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            x = inputs.to(config.DEVICE)
            encoder_input, pos, mask = detr_tools.nested_tensor_to_bb_emb(x, detr)
            encoder_output = detr_tools.bb_emb_to_enc_emb(encoder_input, detr, mask, pos)
            decoder_output = detr_tools.enc_emb_to_dec_emb(encoder_output, detr, mask, pos)
            detr_output = detr_tools.dec_emb_to_detr_out(decoder_output, detr)
            recon = model(detr_output["pred_logits"], detr_output["pred_boxes"])
            loss = F.mse_loss(input=recon, target= decoder_output[-1].transpose(0,1))
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
        if run:
            run["validation/loss"].append(sum_loss / num_inputs)
    return sum_loss / num_inputs
