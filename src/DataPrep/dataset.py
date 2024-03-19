from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from src.ConstantsConfigs.constants import DECODE_TOPIC


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, task: str):
        """
        Text Classification Dataset constructor

        :param df: data
        :param tokenizer: tokenizer for texts
        """
        self.texts = df['text'].values
        self.labels = df['topic'].values
        self.tokenizer = tokenizer
        self.task = task

    def __getitem__(self, idx: int) -> dict:
        """
        Get tokenized text and labels for train

        :param idx: index of data
        :return: tuple (input_ids, attention_mask, label)
        """
        text, label = self.get_row_data(idx)

        text_tokenized = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return {
            'input_ids': text_tokenized.data['input_ids'].squeeze(0),
            'attention_mask': text_tokenized.data['attention_mask'].squeeze(0),
            'label': torch.tensor(self.label2ind(str(label))),
        }

    def __len__(self) -> int:
        """
        Get length of dataset
        :return: length of dataset
        """
        return len(self.texts)

    def get_row_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get row data from df
        :param idx: index of row data
        :return: text and label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label

    def label2ind(self, label: str):
        """
        Convert label to index
        :param label: label data
        :return: int label
        """
        return DECODE_TOPIC[self.task][label]
