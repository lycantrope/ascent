import inspect
import os
import os.path as osp
import shutil
import sys
import tempfile
import types
from importlib import import_module

def load_config(config_path):
    """Load a config file and convert it to a dictionary."""
    filename = osp.abspath(osp.expanduser(config_path))
    fileExtname = osp.splitext(filename)[1]
    assert fileExtname == ".py", "Only py type are supported now!"
    with tempfile.TemporaryDirectory() as temp_config_dir:
        with tempfile.NamedTemporaryFile(
            dir=temp_config_dir, suffix=fileExtname
        ) as temp_config_file:
            temp_config_name = osp.basename(temp_config_file.name)

        # copy file
        shutil.copyfile(filename, osp.join(temp_config_dir, temp_config_name))
        temp_module_name = osp.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
            and not isinstance(value, types.ModuleType)
            and not isinstance(value, types.FunctionType)
            and not inspect.isclass(value)
        }
        # delete imported module
        del sys.modules[temp_module_name]

        # delete temporary file
        os.remove(osp.join(temp_config_dir, temp_config_name))

    return cfg_dict

def to_device(data, device):
    """
    Recursively moves data to the specified device. Supports tensors, lists, tuples, and dictionaries.

    Args:
        data: The input data to be moved to the device.
        device (torch.device): The target device to move the data to.

    Returns:
        The data moved to the specified device.

    Notes:
    - This function is used to move data to the appropriate device before passing it to the model.
    - It is a general utility function that can be used in various contexts.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data