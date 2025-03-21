import config
import torch
from torch.functional import F
import tools.detr_utils as detr_tools

def eval_inverse_detector(model, detr, dataloader):
    model.eval(), detr.eval()
    sum_loss, num_inputs = 0, 0
    with torch.no_grad():
        for batch_id, (inputs, _) in enumerate(dataloader):
            x = inputs.to(config.DEVICE)
            encoder_input, pos, mask = detr_tools.backbone_input_to_encoder_input (x, detr)
            encoder_output = detr_tools.encoder_input_to_encoder_output(encoder_input, detr, mask, pos)
            decoder_output = detr_tools.encoder_output_to_decoder_output(encoder_output, detr, mask, pos)
            detr_output = detr_tools.decoder_output_to_detr_output(decoder_output, detr)   
            recon = model(detr_output["pred_logits"], detr_output["pred_boxes"])
            loss = F.mse_loss(input=recon, target= decoder_output[-1].transpose(0,1))
            sum_loss += loss * len(inputs.tensors)
            num_inputs += len(inputs.tensors)
    return sum_loss / num_inputs
