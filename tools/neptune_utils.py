import torch
import config
from copy import deepcopy
import uuid
import neptune
import yaml
from modules.detr.hubconf import detr_resnet50
from tools.misc_utils import get_module_str_from_model, get_module_str_from_module, generate_tmp_path, ensure_list
import timm
import ast
import modules.inv_resnet.models as inv_resnet_module
import modules.inv_vit_enc.models as inv_vit_enc_module
import modules.inv_vit_bb.models as inv_vit_bb_module
import modules.inv_swin.models as inv_swin_module
import wandb
import shutil
import json
from pathlib import Path


import torch.nn as nn


def parse_tuple(value):
    """Safely parse a string representing a tuple of numbers (ints or floats)."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
                return parsed
        except (ValueError, SyntaxError):
            pass  # Return the original value if parsing fails
    return value


def init_optim(model, module_type: str, optim_state=None, *args, **kwargs):
    optim = getattr(torch.optim, module_type)(params=model.parameters(), *args, **kwargs)
    if optim_state:
        optim.load_state_dict(optim_state)
    return optim


def init_optim_from_params(model, params, optim_state=None):
    module_str = get_module_str_from_model(model)
    if module_str in params['optim_configs']:
        return init_optim(model, optim_state=optim_state, **params['optim_configs'][module_str])


def init_optims_from_params(models, params, optim_states=None):
    optims = []
    for i in range(len(models)):
        optim = init_optim_from_params(models[i], params, optim_state=None if optim_states is None else optim_states[i])
        optims.append(optim)
    return optims


def init_module(module, module_type: str, *args, **kwargs):
    parsed_args = [parse_tuple(arg) for arg in args]
    parsed_kwargs = {k: parse_tuple(v) for k, v in kwargs.items()}
    return getattr(module, module_type)(*parsed_args, **parsed_kwargs)


def init_model(module, module_type, model_state=None, device=config.DEVICE, *args, **kwargs):
    model = init_module(module, module_type, *args, **kwargs).to(device)
    if model_state:
        model.load_state_dict(model_state)
    return model


def upload_checkpoint(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    tmp_path = generate_tmp_path()
    models, optims = ensure_list(models), ensure_list(optims)
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        'model_states': {
            get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()} for model
            in models},
        'optim_states': {
            get_module_str_from_model(model): {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                               optim.state_dict().items()} for model, optim in
            zip(models, optims)},
        'best_loss': deepcopy(best_loss),
        'train_step': train_step,
        'test_step': test_step
    }
    torch.save(checkpoint_dict, tmp_path)
    run['checkpoints'][checkpoint_id].upload(str(tmp_path), wait=True)
    tmp_path.unlink()
    if delete_old_checkpoints:
        delete_checkpoints(except_for=[checkpoint_id], run=run)


def upload_checkpoint_keys(models, optims, best_loss, run, test_step, train_step, keys, delete_old_checkpoints=True):
    tmp_path = generate_tmp_path()
    models, optims = ensure_list(models), ensure_list(optims)
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        'model_states': {keys[i]: {key: value.cpu() for key, value in models[i].state_dict().items()} for i in
                         range(len(models))},
        'optim_states': {keys[i]: {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                   optims[i].state_dict().items()} for i in range(len(optims))},
        'best_loss': deepcopy(best_loss),
        'train_step': train_step,
        'test_step': test_step
    }
    torch.save(checkpoint_dict, tmp_path)
    run['checkpoints'][checkpoint_id].upload(str(tmp_path), wait=True)
    tmp_path.unlink()
    if delete_old_checkpoints:
        delete_checkpoints(except_for=[checkpoint_id], run=run)


def upload_checkpoint_classic(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    tmp_path = generate_tmp_path()
    models, optims = ensure_list(models), ensure_list(optims)
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        'model_states': {
            'bb': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()} for
                   model in models[0]},
            'enc': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()}
                    for model in models[1]},
            'dec': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()}
                    for model in models[2]},
            'detect': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()}
                       for model in models[3]}},
        'optim_states': {'bb': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                optims[0].state_dict().items()},
                         'enc': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                 optims[1].state_dict().items()},
                         'dec': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                 optims[2].state_dict().items()},
                         'detect': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                    optims[3].state_dict().items()}},
        'best_loss': best_loss,
        'train_step': train_step,
        'test_step': test_step
    }
    torch.save(checkpoint_dict, tmp_path)
    run['checkpoints'][checkpoint_id].upload(str(tmp_path), wait=True)
    tmp_path.unlink()
    if delete_old_checkpoints:
        delete_checkpoints(except_for=[checkpoint_id], run=run)


def upload_checkpoint_classic_vit(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    tmp_path = generate_tmp_path()
    models, optims = ensure_list(models), ensure_list(optims)
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        'model_states': {
            'bb': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()} for
                   model in models[0]},
            'enc': {get_module_str_from_model(model): {key: value.cpu() for key, value in model.state_dict().items()}
                    for model in models[1]}},
        'optim_states': {'bb': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                optims[0].state_dict().items()},
                         'enc': {key: value.cpu() if torch.is_tensor(value) else value for key, value in
                                 optims[1].state_dict().items()}},
        'best_loss': best_loss,
        'train_step': train_step,
        'test_step': test_step
    }
    torch.save(checkpoint_dict, tmp_path)
    run['checkpoints'][checkpoint_id].upload(str(tmp_path), wait=True)
    tmp_path.unlink()
    if delete_old_checkpoints:
        delete_checkpoints(except_for=[checkpoint_id], run=run)


def delete_checkpoints(except_for, run):
    checkpoint_ids = get_checkpoint_ids(run)
    for checkpoint_id in checkpoint_ids:
        if checkpoint_id not in except_for:
            del run['checkpoints'][checkpoint_id]


def get_checkpoint_ids(run):
    return list(run.get_structure()["checkpoints"].keys())


def get_checkpoint(run_id, checkpoint_id=None, project=config.PROJECT):
    tmp_path = config.TMP_DIR / str(uuid.uuid4())
    run = neptune.init_run(with_id=run_id, project=project, mode='read-only')
    if checkpoint_id is None:
        checkpoint_id = get_most_recent_checkpoint_id(run)
    run['checkpoints'][checkpoint_id].download(str(tmp_path))
    run.stop()
    checkpoint = torch.load(tmp_path, map_location=torch.device(config.DEVICE))
    tmp_path.unlink()
    return checkpoint


def get_most_recent_checkpoint_id(run):
    checkpoint_ids = get_checkpoint_ids(run)
    return sorted(checkpoint_ids, key=int, reverse=True)[0]


def make_local_run_path(run_id, sub_dir=None):
    local_run_path = config.RUNS_DIR / run_id
    local_run_path.mkdir(exist_ok=True)
    if sub_dir is not None:
        local_run_path = config.RUNS_DIR / run_id / sub_dir
        local_run_path.mkdir(exist_ok=True)
    return local_run_path


def get_run_by_name(run_id, entity='ini-tns', project='object-detection-backer'):
    api = wandb.Api()
    return api.runs(f'{entity}/{project}', filters={"display_name": run_id})[0]


def get_params(run_id, project=config.PROJECT, update=False):
    local_run_path = make_local_run_path(run_id)
    local_params_path = local_run_path / "params.yaml"
    if not local_params_path.exists() or update:
        run = get_run_by_name(run_id=run_id)
        params = run.config
        # run = neptune.init_run(with_id=run_id, project=project, mode='read-only')
        # params = run["params"].fetch()
        with open(str(local_params_path), "w") as f:
            yaml.dump(params, f)
    with open(str(local_params_path), 'r') as f:
        params = yaml.safe_load(f)
    return params


def get_model_state(run_id: str, module_id: str, project=config.PROJECT, update=False, sub_dir=None):
    local_run_path = make_local_run_path(run_id, sub_dir)
    local_model_state_path = local_run_path / (module_id + '_model_state.pt')
    if not local_model_state_path.exists() or update:
        run = neptune.init_run(with_id=run_id, project=project, mode='read-only')
        if sub_dir is None:
            run['model_states'][module_id].download(str(local_model_state_path))
        else:
            run['model_states'][sub_dir][module_id].download(str(local_model_state_path))
        run.stop()
    model_state_dict = torch.load(str(local_model_state_path), map_location=torch.device(config.DEVICE))
    return model_state_dict


def get_model_config_from_params(module, params):
    return params['model_configs'][get_module_str_from_module(module)]


def init_model_from_params(module, params=None, model_state=None, device=config.DEVICE, model_config=None):
    if model_config is None:
        model_config = get_model_cfg(params=params, key=module.__name__.split('.')[1])
        # model_config = get_model_config_from_params(module, params)
    if 'as_in' in model_config:
        model = init_model_as_in(module, model_config['as_in'], load_model_state=False, device=device)
    else:
        model = init_model(module=module, model_state=model_state, **model_config, device=device)
    return model


def init_model_as_in(module, as_in_id, load_model_state=False, device=config.DEVICE):
    if as_in_id == 'detr_resnet50':
        return detr_resnet50(pretrained=load_model_state).to(device)
    if as_in_id == 'vit_base_patch16_224':
        return timm.create_model("vit_base_patch16_224", pretrained=True).to(config.DEVICE)
    else:
        return init_model_from_neptune(run_id=as_in_id, module=module, load_model_state=load_model_state).to(device)


def upload_model_state(model, run):
    tmp_path = generate_tmp_path()
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, tmp_path)
    run['model_states'][get_module_str_from_model(model)].upload(str(tmp_path), wait=True)
    tmp_path.unlink()


def upload_model_state_keys(models, run, keys=None):
    for i in range(len(models)):
        upload_model_state_key(model=models[i], run=run, key=str(i) if keys is None else keys[i])


def upload_model_state_key(model, run, key):
    tmp_path = generate_tmp_path()
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, tmp_path)
    run['model_states'][key].upload(str(tmp_path), wait=True)
    tmp_path.unlink()

def save_model_state_from_tuple(model, run, model_state_key):
    torch.save({key: value.cpu() for key, value in model.state_dict().items()},
               config.RUNS_DIR / run.id / model_state_key)
    # run.save(config.RUNS_DIR / run.model_id / model_state_key, policy='now')
    # artifact = wandb.Artifact(f"model_states_{model_state_key}", type="file")
    # artifact.add_file(tmp_path)
    # run.log_artifact(artifact)
    # run['model_states'][model_id][get_module_str_from_model(model)].upload(str(tmp_path), wait=True)
    # tmp_path.unlink()


def save_model_state_tuple(models, run, model_id):
    for i in range(len(models)):
        save_model_state_from_tuple(models[i], run, model_state_key=f'{model_id}_{str(i)}')

def save_checkpoint_tuple(models, optims, best_loss, train_step, val_step, run, model_state_key):

    checkpoint_dict = {
        'model_states': {str(i): {key: value.cpu() for key, value in models[i].state_dict().items()} for i in range(len(models))},
        'optim_state': {key: value.cpu() if torch.is_tensor(value) else value for key, value in optims.state_dict().items()},
        'best_loss': deepcopy(best_loss),
        'train_step': train_step,
        'val_step': val_step
    }
    torch.save(checkpoint_dict, config.RUNS_DIR / run.id / model_state_key)




def upload_model_state_tuple(models, run, model_id):
    for i in range(len(models)):
        upload_model_state_from_tuple(models[i], run, model_state_key=f'{model_id}_{str(i)}')


def upload_model_state_from_tuple(model, run, model_state_key):
    tmp_path = generate_tmp_path()
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, tmp_path)
    run.save(tmp_path, policy='now')
    # artifact = wandb.Artifact(f"model_states_{model_state_key}", type="file")
    # artifact.add_file(tmp_path)
    # run.log_artifact(artifact)
    # run['model_states'][model_id][get_module_str_from_model(model)].upload(str(tmp_path), wait=True)
    tmp_path.unlink()


def upload_model_states(models, run):
    for model in models:
        upload_model_state(model, run)


def init_model_from_neptune(run_id, module, project=config.PROJECT, update=False, load_model_state=True, sub_dir=None):
    params = get_params(run_id=run_id, project=project, update=update)
    model = init_model_from_params(module, params)
    if load_model_state:
        model_state = get_model_state_key(run_id=run_id, key=module.__name__.split('.')[1],  project='object-detection-backer', entity='ini-tns', update=False)
        # model_state = get_model_state(run_id=run_id, module_id=get_module_str_from_module(module), project=project,
        #                               update=update, sub_dir=sub_dir)
        model.load_state_dict(model_state)
    return model


def get_model_cfg(params, key):
    model_cgf = {}
    prefix = f'params_model_configs_{key}'
    for key, value in params.items():
        if key.startswith(prefix):
            model_cgf[key[len(prefix) + 1:]] = value
    return model_cgf


def init_init_classic_parallel_vit_form_wand(run_id, project=config.PROJECT, update=False, load_model_state=True):
    classic_bb_cfg = {'module_type': 'EnhancedVitBackbone'}
    classic_enc_cfg = {'module_type': 'InverseViTEncoder'}

    classic_bb_to_tensor =  init_model_from_params(inv_vit_bb_module, model_config=classic_bb_cfg)
    classic_bb_to_tensor.load_state_dict(get_model_state_key(run_id=run_id, key='bb_inv_vit_bb', project=project, update=update))

    classic_enc_to_tensor = [init_model_from_params(inv_vit_bb_module, model_config=classic_bb_cfg),
                             init_model_from_params(inv_vit_enc_module, model_config=classic_enc_cfg)]
    classic_enc_to_tensor[0].load_state_dict(get_model_state_key(run_id=run_id, key='enc_inv_vit_bb', project=project, update=update))
    classic_enc_to_tensor[1].load_state_dict(get_model_state_key(run_id=run_id, key='enc_inv_vit_enc', project=project, update=update))

    return classic_bb_to_tensor, classic_enc_to_tensor

def init_modular_parallel_vit_form_wand(run_id, project=config.PROJECT, update=False, load_model_state=True):
    inv_bb = init_model_from_neptune_key(run_id, module=inv_vit_bb_module, project=project, update=update, load_model_state=load_model_state, key='inv_vit_bb')
    inv_enc = init_model_from_neptune_key(run_id, module=inv_vit_enc_module, project=project, update=update, load_model_state=load_model_state, key='inv_vit_encoder')
    return inv_bb, inv_enc

def init_modular_sub_vit_form_wand(run_id, project=config.PROJECT, update=False, load_model_state=True):
    inv_bb = init_model_from_neptune_key(run_id, module=inv_vit_bb_module, project=project, update=update, load_model_state=load_model_state, key='inv_vit_bb')
    inv_sub_enc_1 = init_model_from_neptune_key(run_id, module=inv_vit_enc_module, project=project, update=update, load_model_state=load_model_state, key='inv_vit_sub_encoder_1')
    inv_sub_enc_2 = init_model_from_neptune_key(run_id, module=inv_vit_enc_module, project=project, update=update, load_model_state=load_model_state, key='inv_vit_sub_encoder_2')
    return inv_bb, inv_sub_enc_1, inv_sub_enc_2



def init_modular_parallel_swin_form_wand(run_id, project=config.PROJECT, update=False, load_model_state=True):
    keys = ['0', '1', '2', '3', '4']
    models = []
    for key in keys:
        models.append(init_model_from_neptune_key(run_id, module=inv_swin_module, project=project, update=update, load_model_state=load_model_state, key=key))
    return models

def init_thingy_parallel_swin_form_wand(run_id, project=config.PROJECT, update=False, load_model_state=True):
    keys = ['0', '1', '2', '3', '4']
    models = []
    kwargs = {
        '0': {'module_type': 'SwinBackbone'},
        '1': {'module_type': 'InverseSwinTransformerStage', 'dim': 128, 'out_dim': 128,
              'input_resolution': (56, 56), 'output_resolution': (56, 56), 'depth': 2, 'upsample': False,
              'num_heads': 4},
        '2': {'module_type': 'InverseSwinTransformerStage', 'dim': 256, 'out_dim': 128,
              'input_resolution': (28, 28), 'output_resolution': (56, 56), 'depth': 2, 'upsample': True,
              'num_heads': 8},
        '3': {'module_type': 'InverseSwinTransformerStage', 'dim': 512, 'out_dim': 256,
              'input_resolution': (14, 14), 'output_resolution': (28, 28), 'depth': 18, 'upsample': True,
              'num_heads': 16},
        '4': {'module_type': 'InverseSwinTransformerStage', 'dim': 1024, 'out_dim': 512,
              'input_resolution': (7, 7), 'output_resolution': (14, 14), 'depth': 2, 'upsample': True,
              'num_heads': 32}}
    for key in keys:
        models.append(init_model_from_neptune_key(run_id, module=inv_swin_module, project=project, update=update,
                                                  load_model_state=load_model_state, key=key, model_config=kwargs[key]))
    return models

def init_classic_parallel_swin_form_wand(run_id='car4usiy', project=config.PROJECT, update=False, load_model_state=True):
    configs = {
        '0': {'module_type': 'SwinBackbone'},
        '1': {'module_type': 'InverseSwinTransformerStage', 'dim': 128, 'out_dim': 128,
              'input_resolution': (56, 56), 'output_resolution': (56, 56), 'depth': 2, 'upsample': False,
              'num_heads': 4},
        '2': {'module_type': 'InverseSwinTransformerStage', 'dim': 256, 'out_dim': 128,
              'input_resolution': (28, 28), 'output_resolution': (56, 56), 'depth': 2, 'upsample': True,
              'num_heads': 8},
        '3': {'module_type': 'InverseSwinTransformerStage', 'dim': 512, 'out_dim': 256,
              'input_resolution': (14, 14), 'output_resolution': (28, 28), 'depth': 18, 'upsample': True,
              'num_heads': 16},
        '4': {'module_type': 'InverseSwinTransformerStage', 'dim': 1024, 'out_dim': 512,
              'input_resolution': (7, 7), 'output_resolution': (14, 14), 'depth': 2, 'upsample': True,
              'num_heads': 32}}

    zero_to_img = [init_model_from_params(inv_swin_module, model_config=configs['0'])]
    one_to_img =  [init_model_from_params(inv_swin_module, model_config=configs['0']), init_model_from_params(inv_swin_module, model_config=configs['1'])]
    two_to_img =  [init_model_from_params(inv_swin_module, model_config=configs['0']), init_model_from_params(inv_swin_module, model_config=configs['1']),  init_model_from_params(inv_swin_module, model_config=configs['2'])]
    three_to_img = [init_model_from_params(inv_swin_module, model_config=configs['0']), init_model_from_params(inv_swin_module, model_config=configs['1']),  init_model_from_params(inv_swin_module, model_config=configs['2']), init_model_from_params(inv_swin_module, model_config=configs['3'])]
    four_to_img = [init_model_from_params(inv_swin_module, model_config=configs['0']), init_model_from_params(inv_swin_module, model_config=configs['1']),  init_model_from_params(inv_swin_module, model_config=configs['2']),
                   init_model_from_params(inv_swin_module, model_config=configs['3']), init_model_from_params(inv_swin_module, model_config=configs['4'])]

    local_run_path = make_local_run_path(run_id)
    all_models = [zero_to_img, one_to_img, two_to_img, three_to_img, four_to_img]
    for i in range(len(all_models)):
        for j in range(len(all_models[i])):
            local_model_state_path = local_run_path / (str(i)+ '_' + str(len(all_models[i]) - 1 - j))
            model_state_dict = torch.load(str(local_model_state_path), map_location=torch.device(config.DEVICE))
            all_models[i][j].load_state_dict(model_state_dict)
    return all_models



def init_model_from_neptune_key(run_id, module, project=config.PROJECT, update=False, load_model_state=True, key=None, model_config=None):
    params = get_params(run_id=run_id, project=project, update=update)
    if model_config is None:
        model_config = get_model_cfg(params=params, key=key)
    model = init_model_from_params(module, params, model_config=model_config)
    if load_model_state:
        model_state = get_model_state_key(run_id=run_id, key=key, project=project, update=update)
        model.load_state_dict(model_state)
    return model


def prepare_run(run, neptune_params=None):
    (config.RUNS_DIR / run.id).mkdir(exist_ok=True, parents=True)
    if neptune_params is not None:
        with open(str((config.RUNS_DIR / run.id / 'cfg.json')), "w") as f:
            json.dump(neptune_params, f, indent=4)
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    # validation metrics use val/step
    run.define_metric("val/step")
    run.define_metric("val/*", step_metric="val/step")

def init_model_from_neptune_swin(run_id, module, project=config.PROJECT, update=False, load_model_state=True):
    params = get_params(run_id=run_id, project=project, update=update)
    keys = ['0', '1', '2', '3', '4']
    models = []
    kwargs = {
        '0': {'module_type': 'SwinBackbone'},
        '1': {'module_type': 'InverseSwinTransformerStage', 'dim': 128, 'out_dim': 128,
              'input_resolution': (56, 56), 'output_resolution': (56, 56), 'depth': 2, 'upsample': False,
              'num_heads': 4},
        '2': {'module_type': 'InverseSwinTransformerStage', 'dim': 256, 'out_dim': 128,
              'input_resolution': (28, 28), 'output_resolution': (56, 56), 'depth': 2, 'upsample': True,
              'num_heads': 8},
        '3': {'module_type': 'InverseSwinTransformerStage', 'dim': 512, 'out_dim': 256,
              'input_resolution': (14, 14), 'output_resolution': (28, 28), 'depth': 18, 'upsample': True,
              'num_heads': 16},
        '4': {'module_type': 'InverseSwinTransformerStage', 'dim': 1024, 'out_dim': 512,
              'input_resolution': (7, 7), 'output_resolution': (14, 14), 'depth': 2, 'upsample': True,
              'num_heads': 32}}
    for key in keys:
        model_config = kwargs[key]
        model = init_model_from_params(module, params, model_config=model_config)
        if load_model_state:
            model_state = get_model_state_key(run_id=run_id, key=key, project=project, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    return models


def init_model_from_neptune_inverse_resnet(run_id, project=config.PROJECT, update=False,
                                           conc=False, load_model_state=True):
    params = get_params(run_id=run_id, project=project, update=update)
    keys = ['0', '1', '2', '3', '4']
    models = []
    kwargs = {
        '0': {'module_type': 'InverseResnetBlock', 'in_channels': 64, 'out_channels': 3, 'last_output': True},
        '1': {'module_type': 'InverseResnetBlock', 'in_channels': 256, 'out_channels': 64, 'upsample': False},
        '2': {'module_type': 'InverseResnetBlock', 'in_channels': 512},
        '3': {'module_type': 'InverseResnetBlock', 'in_channels': 1024},
        '4': {'module_type': 'InverseResnetBlock', 'in_channels': 2048}}

    for key in keys:
        model_config = kwargs[key]
        model = init_model_from_params(inv_resnet_module, params, model_config=model_config)
        if load_model_state:
            model_state = get_model_state_key(run_id=run_id, key=key, project=project, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    if conc:
        return nn.Sequential(*reversed(models))
    return models


def init_model_from_neptune_inverse_resnet_detr(run_id, project=config.PROJECT, update=False,
                                                conc=False, load_model_state=True):
    params = get_params(run_id=run_id, project=project, update=update)
    keys = ['0', '1', '2', '3', '4', '5']
    models = []
    kwargs = {
        '0': {'module_type': 'InverseResnetBlock', 'in_channels': 64, 'out_channels': 3, 'last_output': True},
        '1': {'module_type': 'InverseResnetBlock', 'in_channels': 256, 'out_channels': 64, 'upsample': False},
        '2': {'module_type': 'InverseResnetBlock', 'in_channels': 512},
        '3': {'module_type': 'InverseResnetBlock', 'in_channels': 1024},
        '4': {'module_type': 'InverseResnetBlock', 'in_channels': 2048},
        '5': {'module_type': 'InverseInputProjection'}}

    for key in keys:
        model_config = kwargs[key]
        model = init_model_from_params(inv_resnet_module, params, model_config=model_config)
        if load_model_state:
            model_state = get_model_state_key(run_id=run_id, key=key, project=project, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    if conc:
        return nn.Sequential(*reversed(models))
    return models


def init_model_from_neptune_sub_vit(run_id, project=config.PROJECT, update=False,
                                    conc=False, load_model_state=True):
    params = get_params(run_id=run_id, project=project, update=update)
    keys = ['inv_vit_bb', 'inv_vit_sub_encoder_1', 'inv_vit_sub_encoder_2']
    models = []
    kwargs = {'inv_vit_bb': {'module_type': 'EnhancedVitBackbone'},
              'inv_vit_sub_encoder_1': {'module_type': 'InverseViTEncoder', 'depth': 6},
              'inv_vit_sub_encoder_2': {'module_type': 'InverseViTEncoder', 'depth': 6}}
    modules = [inv_vit_bb_module, inv_vit_enc_module, inv_vit_enc_module]
    for key, module in zip(keys, modules):
        model_config = kwargs[key]
        model = init_model_from_params(module, params, model_config=model_config)
        if load_model_state:
            model_state = get_model_state_key(run_id=run_id, key=key, project=project, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    if conc:
        return nn.Sequential(*reversed(models))
    return models

def get_artifact_model_state_key(run, key):
    artifact_ids = [a for a in run.logged_artifacts()]
    for artifact_id in artifact_ids:
        if artifact_id.name.startswith(f'model_states_{key}'):
            return artifact_id.name
    raise FileNotFoundError(f'{key} not found')


def get_model_state_key(run_id: str, key: str,  project='object-detection-backer', entity='ini-tns', update=False):
    local_run_path = make_local_run_path(run_id)
    local_model_state_path = local_run_path / (key + '_model_state.pt')
    if not local_model_state_path.exists() or update:
        api = wandb.Api()
        run = get_run_by_name(run_id=run_id)
        artifact_id = get_artifact_model_state_key(run, key)
        artifact = api.artifact(f'{entity}/{project}/{artifact_id}')
        src = artifact.file(local_run_path)
        shutil.copy(src, local_model_state_path)
        Path(src).unlink()
    model_state_dict = torch.load(str(local_model_state_path), map_location=torch.device(config.DEVICE))
    return model_state_dict


def init_from_checkpoint(checkpoint, module, params, device=config.DEVICE):
    module_id = get_module_str_from_module(module)
    model_state = checkpoint['model_states'][module_id]
    model = init_model_from_params(module=module, params=params, model_state=model_state, device=device)
    model.load_state_dict(model_state)
    optim_state = checkpoint['optim_states'][module_id]
    optim = init_optim(model, **params['optim_configs'][module_id], optim_state=optim_state)
    return model, optim


def update_model_configs(params, modules, run_ids):
    for module, run_id in zip(modules, run_ids):
        params['model_configs'][get_module_str_from_module(module)] = \
            get_params(run_id)['model_configs'][get_module_str_from_module(module)]


if __name__ == '__main__':
    init_modular_sub_vit_form_wand(run_id='OB-1457')
    # init_classic_parallel_swin_form_wand()