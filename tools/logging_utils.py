import ast
import json
import shutil
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

import config
from tools.misc_utils import ensure_list, generate_tmp_path, get_module_str_from_model, get_module_str_from_module


def _ensure_runtime_dirs():
    config.RUNS_DIR.mkdir(exist_ok=True, parents=True)
    config.TMP_DIR.mkdir(exist_ok=True, parents=True)


def _resolve_experiment_id(experiment_id=None, run_id=None):
    experiment_id = experiment_id or run_id
    if experiment_id is None:
        raise ValueError("An experiment id is required for this operation")
    return experiment_id


def _to_jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _to_float(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def stringify_unsupported(value):
    return _to_jsonable(value)


def progress(iterable, *args, **kwargs):
    return tqdm(iterable, *args, **kwargs)


class LocalMetric:
    def __init__(self, run, name):
        self.run = run
        self.name = name

    def __setitem__(self, key, value):
        if self.name == "params":
            params = self.run.load_params()
            params[key] = _to_jsonable(value)
            if key == "epochs":
                self.run.flush_epoch_metrics(epoch=value)
            self.run.save_params(params)
            return
        self.run.save_json(f"{self.name}/{key}", value)

    def append(self, value):
        self.run.log_metric(self.name, value)

    def upload(self, src, wait=False):
        self.run.save_file(self.name, src)

    def download(self, dst):
        self.run.load_file(self.name, dst)


class LocalRun:
    def __init__(self, experiment_id=None, mode=None, params=None):
        _ensure_runtime_dirs()
        self.id = experiment_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.mode = mode
        self.path = config.RUNS_DIR / self.id
        self.path.mkdir(exist_ok=True, parents=True)
        self.metrics_path = self.path / "metrics.jsonl"
        self._epoch_metrics = {}
        print(f"Local run directory: {self.path}")
        if params is not None:
            self["params"] = params

    def __getitem__(self, key):
        return LocalMetric(self, key)

    def __setitem__(self, key, value):
        if key == "params":
            self.save_params(value)
        else:
            self.save_json(key, value)

    def save_params(self, params):
        params = _to_jsonable(params)
        with (self.path / "config.json").open("w") as f:
            json.dump(params, f, indent=2)
        with (self.path / "config.yaml").open("w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    def load_params(self):
        for name in ("config.json", "config.yaml"):
            path = self.path / name
            if path.exists():
                with path.open("r") as f:
                    return json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        return {}

    def save_json(self, key, value):
        dst = self.path / (key.replace("/", "_") + ".json")
        dst.parent.mkdir(exist_ok=True, parents=True)
        with dst.open("w") as f:
            json.dump(_to_jsonable(value), f, indent=2)

    def log_metric(self, name, value):
        if name == "train/step":
            return
        if name.startswith("train/"):
            numeric_value = _to_float(value)
            if numeric_value is not None:
                bucket = self._epoch_metrics.setdefault(name, [])
                bucket.append(numeric_value)
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "metric": name,
            "value": _to_jsonable(value),
        }
        self.write_metric(record)

    def write_metric(self, record):
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(_to_jsonable(record)) + "\n")

    def flush_epoch_metrics(self, epoch=None):
        if not self._epoch_metrics:
            return
        for name, values in sorted(self._epoch_metrics.items()):
            if not values:
                continue
            self.write_metric({
                "time": datetime.now().isoformat(timespec="seconds"),
                "epoch": _to_jsonable(epoch),
                "metric": name,
                "value": sum(values) / len(values),
                "count": len(values),
            })
        self._epoch_metrics.clear()

    def save_file(self, key, src):
        dst = self.path / key
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(src, dst)

    def load_file(self, key, dst):
        src = self.path / key
        if not src.exists():
            raise FileNotFoundError(f"Could not find artifact '{key}' in {self.path}")
        shutil.copy2(src, dst)

    def get_structure(self):
        checkpoints = self.path / "checkpoints"
        return {"checkpoints": {p.name: None for p in checkpoints.iterdir()}} if checkpoints.exists() else {
            "checkpoints": {}
        }

    def stop(self):
        self.flush_epoch_metrics()

    def define_metric(self, *args, **kwargs):
        pass


def init_run(experiment_id=None, with_id=None, project=None, mode=None, source_files=None, **kwargs):
    del project, source_files, kwargs
    return LocalRun(experiment_id=with_id or experiment_id, mode=mode)


def parse_tuple(value):
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return value


def init_optim(model, module_type: str, optim_state=None, *args, **kwargs):
    optim = getattr(torch.optim, module_type)(params=model.parameters(), *args, **kwargs)
    if optim_state:
        optim.load_state_dict(optim_state)
    return optim


CONFIG_KEY_ALIASES = {
    "inv_detr_bb": ("inv_detr_bb", "inv_bb", "inverse_backbone"),
    "inv_detr_enc": ("inv_detr_enc", "inv_enc", "inverse_encoder"),
    "inv_detr_dec": ("inv_detr_dec", "inv_dec", "inverse_decoder"),
    "inv_detr_pred": ("inv_detr_pred", "inv_detect", "inverse_detector"),
    "inv_vit_enc": ("inv_vit_enc", "inv_vit_encoder", "inv_enc"),
    "inv_vit_bb": ("inv_vit_bb", "inv_bb"),
}


def _candidate_config_keys(key):
    return CONFIG_KEY_ALIASES.get(key, (key,))


def _get_config(configs, key):
    for candidate in _candidate_config_keys(key):
        if candidate in configs:
            return configs[candidate], candidate
    return None, key


def init_optim_from_params(model, params, optim_state=None):
    module_str = get_module_str_from_model(model)
    optim_config, config_key = _get_config(params["optim_configs"], module_str)
    if optim_config is None:
        available = ", ".join(sorted(params["optim_configs"].keys()))
        raise KeyError(f"No optimizer config found for '{module_str}'. Available configs: {available}")
    return init_optim(model, optim_state=optim_state, **optim_config)


def init_optims_from_params(models, params, optim_states=None):
    return [
        init_optim_from_params(model, params, optim_state=None if optim_states is None else optim_states[i])
        for i, model in enumerate(models)
    ]


def init_module(module, module_type: str, *args, **kwargs):
    parsed_args = [parse_tuple(arg) for arg in args]
    parsed_kwargs = {k: parse_tuple(v) for k, v in kwargs.items()}
    return getattr(module, module_type)(*parsed_args, **parsed_kwargs)


def init_model(module, module_type, model_state=None, device=config.DEVICE, *args, **kwargs):
    model = init_module(module, module_type, *args, **kwargs).to(device)
    if model_state:
        model.load_state_dict(model_state)
    return model


def _state_dict_cpu(model):
    model = getattr(model, "_orig_mod", model)
    return {key: value.cpu() for key, value in model.state_dict().items()}


def _optim_state_cpu(optim):
    if optim is None:
        raise ValueError("Cannot checkpoint a None optimizer. Check optimizer config keys for this model.")
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in optim.state_dict().items()}


def upload_checkpoint(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    models, optims = ensure_list(models), ensure_list(optims)
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        "model_states": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models},
        "optim_states": {
            get_module_str_from_model(model): _optim_state_cpu(optim) for model, optim in zip(models, optims)
        },
        "best_loss": deepcopy(best_loss),
        "train_step": train_step,
        "test_step": test_step,
    }
    _save_checkpoint_dict(checkpoint_dict, run, checkpoint_id, delete_old_checkpoints)


def upload_checkpoint_keys(models, optims, best_loss, run, test_step, train_step, keys, delete_old_checkpoints=True):
    checkpoint_id = str(test_step).zfill(5)
    optim_states = (
        {keys[i]: _optim_state_cpu(optims[i]) for i in range(len(optims))}
        if isinstance(optims, (list, tuple))
        else {"shared": _optim_state_cpu(optims)}
    )
    checkpoint_dict = {
        "model_states": {keys[i]: _state_dict_cpu(models[i]) for i in range(len(models))},
        "optim_states": optim_states,
        "best_loss": deepcopy(best_loss),
        "train_step": train_step,
        "test_step": test_step,
    }
    _save_checkpoint_dict(checkpoint_dict, run, checkpoint_id, delete_old_checkpoints)


def upload_checkpoint_classic(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        "model_states": {
            "bb": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[0]},
            "enc": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[1]},
            "dec": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[2]},
            "detect": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[3]},
        },
        "optim_states": {
            "bb": _optim_state_cpu(optims[0]),
            "enc": _optim_state_cpu(optims[1]),
            "dec": _optim_state_cpu(optims[2]),
            "detect": _optim_state_cpu(optims[3]),
        },
        "best_loss": best_loss,
        "train_step": train_step,
        "test_step": test_step,
    }
    _save_checkpoint_dict(checkpoint_dict, run, checkpoint_id, delete_old_checkpoints)


def upload_checkpoint_classic_vit(models, optims, best_loss, run, test_step, train_step, delete_old_checkpoints=True):
    checkpoint_id = str(test_step).zfill(5)
    checkpoint_dict = {
        "model_states": {
            "bb": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[0]},
            "enc": {get_module_str_from_model(model): _state_dict_cpu(model) for model in models[1]},
        },
        "optim_states": {"bb": _optim_state_cpu(optims[0]), "enc": _optim_state_cpu(optims[1])},
        "best_loss": best_loss,
        "train_step": train_step,
        "test_step": test_step,
    }
    _save_checkpoint_dict(checkpoint_dict, run, checkpoint_id, delete_old_checkpoints)


def _save_checkpoint_dict(checkpoint_dict, run, checkpoint_id, delete_old_checkpoints=True):
    path = run.path / "checkpoints" / checkpoint_id
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint_dict, path)
    if delete_old_checkpoints:
        delete_checkpoints(except_for=[checkpoint_id], run=run)


def delete_checkpoints(except_for, run):
    checkpoint_dir = run.path / "checkpoints"
    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.iterdir():
        if path.name not in except_for:
            path.unlink()


def get_checkpoint_ids(run):
    checkpoint_dir = run.path / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return [path.name for path in checkpoint_dir.iterdir()]


def get_checkpoint(experiment_id=None, checkpoint_id=None, project=None, run_id=None):
    del project
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    run = init_run(with_id=experiment_id, mode="read-only")
    if checkpoint_id is None:
        checkpoint_id = get_most_recent_checkpoint_id(run)
    checkpoint_path = run.path / "checkpoints" / checkpoint_id
    return torch.load(checkpoint_path, map_location=torch.device(config.DEVICE))


def get_most_recent_checkpoint_id(run):
    checkpoint_ids = get_checkpoint_ids(run)
    if not checkpoint_ids:
        raise FileNotFoundError(f"No checkpoints found in {run.path / 'checkpoints'}")
    return sorted(checkpoint_ids, key=int, reverse=True)[0]


def make_local_run_path(experiment_id=None, sub_dir=None, run_id=None):
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    local_run_path = config.RUNS_DIR / experiment_id
    local_run_path.mkdir(exist_ok=True, parents=True)
    if sub_dir is not None:
        local_run_path = local_run_path / sub_dir
        local_run_path.mkdir(exist_ok=True, parents=True)
    return local_run_path


def get_params(experiment_id=None, project=None, update=False, run_id=None):
    del project, update
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    local_run_path = make_local_run_path(experiment_id)
    for name in ("config.yaml", "config.json", "params.yaml"):
        path = local_run_path / name
        if path.exists():
            with path.open("r") as f:
                return yaml.safe_load(f) if path.suffix in (".yaml", ".yml") else json.load(f)
    raise FileNotFoundError(f"No config file found in {local_run_path}")


def get_model_state(experiment_id=None, module_id=None, project=None, update=False, sub_dir=None, run_id=None):
    del project, update
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    local_run_path = make_local_run_path(experiment_id, sub_dir)
    local_model_state_path = local_run_path / "model_states" / module_id
    if not local_model_state_path.exists():
        local_model_state_path = local_run_path / (module_id + "_model_state.pt")
    return torch.load(str(local_model_state_path), map_location=torch.device(config.DEVICE))


def get_model_config_from_params(module, params):
    module_str = get_module_str_from_module(module)
    model_config, _ = _get_config(params["model_configs"], module_str)
    if model_config is None:
        available = ", ".join(sorted(params["model_configs"].keys()))
        raise KeyError(f"No model config found for '{module_str}'. Available configs: {available}")
    return model_config


def get_model_cfg(params, key):
    if "model_configs" in params:
        model_config, _ = _get_config(params["model_configs"], key)
        if model_config is not None:
            return model_config
    model_cfg = {}
    prefix = f"params_model_configs_{key}"
    for cfg_key, value in params.items():
        if cfg_key.startswith(prefix):
            model_cfg[cfg_key[len(prefix) + 1:]] = value
    return model_cfg


def init_model_from_params(module, params=None, model_state=None, device=config.DEVICE, model_config=None):
    if model_config is None:
        model_config = get_model_cfg(params=params, key=get_module_str_from_module(module))
    if "as_in" in model_config:
        model = init_model_as_in(module, model_config["as_in"], load_model_state=False, device=device)
    else:
        model = init_model(module=module, model_state=model_state, **model_config, device=device)
    return model


def init_model_as_in(module, as_in_id, load_model_state=False, device=config.DEVICE):
    if as_in_id == "detr_resnet50":
        from modules.detr.hubconf import detr_resnet50

        return detr_resnet50(pretrained=load_model_state).to(device)
    if as_in_id == "vit_base_patch16_224":
        import timm

        return timm.create_model("vit_base_patch16_224", pretrained=True).to(config.DEVICE)
    return init_model_from_run(experiment_id=as_in_id, module=module, load_model_state=load_model_state).to(device)


def upload_model_state(model, run):
    upload_model_state_key(model, run, get_module_str_from_model(model))


def upload_model_state_keys(models, run, keys=None):
    for i in range(len(models)):
        upload_model_state_key(model=models[i], run=run, key=str(i) if keys is None else keys[i])


def upload_model_state_key(model, run, key):
    dst = run.path / "model_states" / key
    dst.parent.mkdir(exist_ok=True, parents=True)
    torch.save(_state_dict_cpu(model), dst)


def save_model_state_from_tuple(model, run, model_state_key):
    upload_model_state_key(model, run, model_state_key)


def save_model_state_tuple(models, run, model_id):
    for i in range(len(models)):
        save_model_state_from_tuple(models[i], run, model_state_key=f"{model_id}_{str(i)}")


def save_checkpoint_tuple(models, optims, best_loss, train_step, val_step, run, model_state_key):
    checkpoint_dict = {
        "model_states": {str(i): _state_dict_cpu(models[i]) for i in range(len(models))},
        "optim_state": _optim_state_cpu(optims),
        "best_loss": deepcopy(best_loss),
        "train_step": train_step,
        "val_step": val_step,
    }
    checkpoint_dir = run.path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint_dict, checkpoint_dir / model_state_key)


def upload_model_state_tuple(models, run, model_id):
    for i in range(len(models)):
        upload_model_state_from_tuple(models[i], run, model_state_key=f"{model_id}_{str(i)}")


def upload_model_state_from_tuple(model, run, model_state_key):
    upload_model_state_key(model, run, model_state_key)


def upload_model_states(models, run):
    for model in models:
        upload_model_state(model, run)


def init_model_from_run(
    experiment_id=None, module=None, project=None, update=False, load_model_state=True, sub_dir=None, run_id=None
):
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    params = get_params(experiment_id=experiment_id, project=project, update=update)
    model = init_model_from_params(module, params)
    if load_model_state:
        model_state = get_model_state_key(
            experiment_id=experiment_id, key=get_module_str_from_module(module), update=False
        )
        model.load_state_dict(model_state)
    return model


def init_model_from_run_key(
    experiment_id=None,
    module=None,
    project=None,
    update=False,
    load_model_state=True,
    key=None,
    model_config=None,
    run_id=None,
):
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    params = get_params(experiment_id=experiment_id, project=project, update=update)
    if model_config is None:
        model_config = get_model_cfg(params=params, key=key)
    model = init_model_from_params(module, params, model_config=model_config)
    if load_model_state:
        model_state = get_model_state_key(experiment_id=experiment_id, key=key, project=project, update=update)
        model.load_state_dict(model_state)
    return model


def prepare_run(run, run_params=None):
    run.path.mkdir(exist_ok=True, parents=True)
    if run_params is not None:
        run["params"] = run_params


def init_model_from_run_swin(
    experiment_id=None, module=None, project=None, update=False, load_model_state=True, run_id=None
):
    del project
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    keys = ["0", "1", "2", "3", "4"]
    kwargs = {
        "0": {"module_type": "SwinBackbone"},
        "1": {
            "module_type": "InverseSwinTransformerStage",
            "dim": 128,
            "out_dim": 128,
            "input_resolution": (56, 56),
            "output_resolution": (56, 56),
            "depth": 2,
            "upsample": False,
            "num_heads": 4,
        },
        "2": {
            "module_type": "InverseSwinTransformerStage",
            "dim": 256,
            "out_dim": 128,
            "input_resolution": (28, 28),
            "output_resolution": (56, 56),
            "depth": 2,
            "upsample": True,
            "num_heads": 8,
        },
        "3": {
            "module_type": "InverseSwinTransformerStage",
            "dim": 512,
            "out_dim": 256,
            "input_resolution": (14, 14),
            "output_resolution": (28, 28),
            "depth": 18,
            "upsample": True,
            "num_heads": 16,
        },
        "4": {
            "module_type": "InverseSwinTransformerStage",
            "dim": 1024,
            "out_dim": 512,
            "input_resolution": (7, 7),
            "output_resolution": (14, 14),
            "depth": 2,
            "upsample": True,
            "num_heads": 32,
        },
    }
    models = []
    for key in keys:
        model = init_model_from_params(module, params={}, model_config=kwargs[key])
        if load_model_state:
            model_state = get_model_state_key(experiment_id=experiment_id, key=key, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    return models


def init_model_from_run_sub_vit(
    experiment_id=None, project=None, update=False, conc=False, load_model_state=True, run_id=None
):
    del project
    import modules.inv_vit.inv_bb.models as inv_vit_bb_module
    import modules.inv_vit.inv_enc.models as inv_vit_enc_module

    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    keys = ["inv_vit_bb", "inv_vit_sub_encoder_1", "inv_vit_sub_encoder_2"]
    kwargs = {
        "inv_vit_bb": {"module_type": "EnhancedVitBackbone"},
        "inv_vit_sub_encoder_1": {"module_type": "InverseViTEncoder", "depth": 6},
        "inv_vit_sub_encoder_2": {"module_type": "InverseViTEncoder", "depth": 6},
    }
    modules = [inv_vit_bb_module, inv_vit_enc_module, inv_vit_enc_module]
    models = []
    for key, module in zip(keys, modules):
        model = init_model_from_params(module, params={}, model_config=kwargs[key])
        if load_model_state:
            model_state = get_model_state_key(experiment_id=experiment_id, key=key, update=update)
            model.load_state_dict(model_state)
        models.append(model)
    return nn.Sequential(*reversed(models)) if conc else models


def get_model_state_key(experiment_id=None, key=None, project=None, entity=None, update=False, run_id=None):
    del project, entity, update
    experiment_id = _resolve_experiment_id(experiment_id, run_id)
    local_run_path = make_local_run_path(experiment_id)
    candidates = [
        local_run_path / "model_states" / key,
        local_run_path / (key + "_model_state.pt"),
        local_run_path / key,
    ]
    for path in candidates:
        if path.exists():
            return torch.load(str(path), map_location=torch.device(config.DEVICE))
    raise FileNotFoundError(f"No model state for key '{key}' found in {local_run_path}")


def init_from_checkpoint(checkpoint, module, params, device=config.DEVICE):
    module_id = get_module_str_from_module(module)
    model_state = checkpoint["model_states"][module_id]
    model = init_model_from_params(module=module, params=params, model_state=model_state, device=device)
    model.load_state_dict(model_state)
    optim_state = checkpoint["optim_states"][module_id]
    optim = init_optim(model, **params["optim_configs"][module_id], optim_state=optim_state)
    return model, optim


def update_model_configs(params, modules, experiment_ids=None, run_ids=None):
    experiment_ids = experiment_ids or run_ids or []
    for module, experiment_id in zip(modules, experiment_ids):
        if experiment_id is None:
            continue
        params["model_configs"][get_module_str_from_module(module)] = get_params(experiment_id)[
            "model_configs"
        ][get_module_str_from_module(module)]
