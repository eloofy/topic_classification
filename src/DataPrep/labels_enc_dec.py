from typing import Dict


def ids_to_str_class(labels: Dict):
    """
    Ids to str class dict

    :param labels: all labels dictionary
    :return: idx to class dict
    """
    dict_idx_to_class = {}
    for name_cls, idx in labels.items():
        dict_idx_to_class[idx] = name_cls

    return dict_idx_to_class
