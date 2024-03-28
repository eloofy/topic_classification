from pathlib import Path
from typing import Dict

import streamlit as st
import torch
from transformers import AutoTokenizer

from src.ConstantsConfigs.config import ExperimentConfig
from src.ConstantsConfigs.constants import DEFAULT_PROJECT_PATH
from src.DataPrep.labels_enc_dec import ids_to_str_class
from src.NN.nn_model import BERTModelClassic


class TextClassifierApp:
    def __init__(self, model_path: Path, cfg: ExperimentConfig, labels: Dict):
        """

        :param model_path: best model path
        :param cfg: cfg best model
        :param labels: labels dictionary
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.model = self.load_model(DEFAULT_PROJECT_PATH / model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.data_config.pretrained_tokenizer,
        )
        self.model.eval()
        self.labels = ids_to_str_class(labels)

    def load_model(self, model_path: Path) -> BERTModelClassic:
        """

        :param model_path: best model path
        :return: best model class
        """
        model = BERTModelClassic(cfg=self.cfg.module_config)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)['state_dict'],
        )

        return model.to(self.device)

    def predict(self, text: str):
        """

        :param text: text to predict
        :return: prediction and output probs
        """
        tokenized_text = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt',
            return_attention_mask=True,
        ).to(self.device)

        with torch.no_grad():
            output = self.model.predict_step(tokenized_text.data, 0, 0)
            prediction = torch.argmax(output).item()
        return prediction, output

    def run(self):
        """
        Run the app
        :return:
        """
        st.title('Классификация темы соц/дем')
        st.write('Новая модель (90кк++ params) + new data')

        user_input = st.text_area('Введите текст для классификации:', '')

        if st.button('Классифицировать'):
            if user_input.strip():
                prediction, output = self.predict(user_input)
                st.success(f'Класс текста: {self.labels[prediction]}')
            else:
                st.warning('Пожалуйста, введите текст для классификации.')
