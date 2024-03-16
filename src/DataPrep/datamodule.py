from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.ConstantsConfigs.config import DataConfig
from src.ConstantsConfigs.constants import DEFAULT_PROJECT_PATH
from src.DataPrep.dataset import TextClassificationDataset
from src.DataPrep.load_data_df import load_dataset

DEFAULT_DATA_PATH = Path(DEFAULT_PROJECT_PATH / 'dataset')


class TextClassificationDatamodule(LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ) -> None:
        """
        Constructor for TextClassificationDataset class

        :param cfg: data configuration
        """
        super().__init__()
        self.cfg = cfg

        self.data_path_file = Path(DEFAULT_DATA_PATH / cfg.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_tokenizer)

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[TextClassificationDataset] = None
        self.data_val: Optional[TextClassificationDataset] = None
        self.data_test: Optional[TextClassificationDataset] = None

    def setup(self, stage: str) -> None:
        """
        Setup data processing

        :param stage: stage of data loading
        :return:
        """
        if stage == 'fit':
            data_full = load_dataset(self.data_path_file)
            data_train, data_val = train_test_split(
                data_full,
                test_size=self.cfg.train_size,
                stratify=data_full['topic'],
            )
            self.data_train = TextClassificationDataset(
                data_train,
                tokenizer=self.tokenizer,
                task=self.cfg.task_name,
            )
            self.data_val = TextClassificationDataset(
                data_train,
                tokenizer=self.tokenizer,
                task=self.cfg.task_name,
            )

        elif stage == 'test':
            return

    def train_dataloader(self):
        """
        Load training dataloader
        :return: training dataloader
        """
        return DataLoader(
            self.data_train,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Load validation dataloader
        :return: validation dataloader
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Load test dataloader
        :return: test dataloader
        """
        return DataLoader
