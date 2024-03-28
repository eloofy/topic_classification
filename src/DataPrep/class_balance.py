from collections import Counter

import torch


def get_class_weights(labels_data):
    """
    Get class weights for each class

    :param labels_data: all labels
    :return: weights
    """
    dict_counts = dict(
        sorted(
            Counter(labels_data).items(),
            key=lambda x: x[0],
        ),
    )
    weights = [1 / (len(dict_counts) * x) for x in dict_counts.values()]

    return torch.tensor(weights)
