from pathlib import Path

import pandas as pd


def load_dataset(path_dataset: Path) -> pd.DataFrame:
    """
    Load dataset
    :param path_dataset: path to dataset
    :return: dataset data
    """
    return pd.read_excel(path_dataset, index_col=0)
