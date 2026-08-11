
import torch
import timm
from tools.logging_utils import init_model_from_run, init_model_from_run_swin
import modules.inv_swin.models as inv_swin_module


def tensor_to_embs(tensor, model):
    """Return image, patch, stage, and head representations from a Swin model."""
    embs = [tensor, model.patch_embed(tensor)]
    for layer in model.layers:
        embs.append(layer(embs[-1]))
    embs.append(model.forward_head(model.norm(embs[-1])))
    return embs


def chain_invert(emb, inv_networks):
    """Apply inverse networks in reverse order to reconstruct an earlier representation."""
    for inv_network in reversed(inv_networks):
        emb = inv_network(emb)
    return emb


def invert_embs(embs, inv_networks):
    """Invert each Swin representation with its corresponding local inverse module."""
    inverted_embs = []
    for i in range(len(inv_networks)):
        inverted_embs.append(inv_networks[i](embs[i+1]))
    return inverted_embs


def invert_embs_to_imgs(embs, inv_networks):
    """Reconstruct images by chaining inverse modules from each representation level."""
    recons = []
    for i in range(len(embs)):
        emb = embs[i]
        for j in range(i, -1, -1):
            emb = inv_networks[j](emb)
        recons.append(emb)
    return recons


# Misc
def init_modules(run_id):
    """Load Swin inverse modules from a local run id."""
    models = init_model_from_run_swin(run_id=run_id, module=inv_swin_module)
    return models


if __name__ == '__main__':
    model = timm.create_model(model_name='swin_base_patch4_window7_224')
    a =  torch.rand(size=(10, 3, 224, 224))
    int_embs = tensor_to_embs(tensor=a, model=model)
