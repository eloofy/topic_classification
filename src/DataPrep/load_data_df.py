import pandas as pd
from pathlib import Path


def load_dataset(path_dataset: Path) -> pd.DataFrame:
    """
    Load dataset
    :param path_dataset: path to dataset
    :return: dataset data
    """
    df = pd.read_excel(path_dataset, index_col=0)

    return df
