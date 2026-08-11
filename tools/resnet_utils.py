import torch
import timm
from tools.logging_utils import init_model_from_run, init_model_from_run_inverse_resnet_detr


def tensor_to_embs(tensor, model, input_proj=None):
    int_reps = [tensor, model.maxpool(model.act1(model.bn1(model.conv1(tensor))))]
    int_reps.append(model.layer1(int_reps[-1]))
    int_reps.append(model.layer2(int_reps[-1]))
    int_reps.append(model.layer3(int_reps[-1]))
    int_reps.append(model.layer4(int_reps[-1]))
    if input_proj:
        int_reps.append(input_proj(int_reps[-1]))
    return int_reps


def chain_invert(emb, inv_networks):
    for inv_network in reversed(inv_networks):
        emb = inv_network(emb)
    return emb


def invert_embs(embs, inv_networks):
    inverted_embs = []
    for i in range(len(inv_networks)):
        inverted_embs.append(inv_networks[i](embs[i+1]))
    return inverted_embs


def invert_embs_to_imgs(embs, inv_networks):
    recons = []
    for i in range(len(embs)):
        emb = embs[i]
        for j in range(i, -1, -1):
            emb = inv_networks[j](emb)
        recons.append(emb)
    return recons


# Misc
def init_modules(run_id, *args, **kwargs):
    models = init_model_from_run_inverse_resnet_detr(run_id=run_id, *args, **kwargs)
    return models


if __name__ == '__main__':
    model = timm.create_model(model_name='swin_base_patch4_window7_224')
    a =  torch.rand(size=(10, 3, 224, 224))
    int_embs = tensor_to_embs(tensor=a, swin=model)