import uuid
import config


def get_parent_file(path):
    return str(path.parent).rsplit('/', 1)[-1]


def get_module_str_from_model(model):
    split = model.__module__.split('.')
    if split[0] == 'timm':
        return split[-1]
    else:
        return split[1]


def get_module_str_from_module(module):
    return module.__name__.split('.')[1]


def generate_tmp_path():
    return config.TMP_DIR / str(uuid.uuid4())


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return obj
    elif obj is None:
        return []
    else:
        return [obj]
