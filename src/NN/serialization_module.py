import importlib
from typing import Any


def load_object(obj_path: str) -> Any:
    """
    Load an object from a file
    :param obj_path: path to the file
    :return: object loaded from the file
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obg_path = obj_path_list.pop(0)

    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obg_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)
