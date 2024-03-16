import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.ConstantsConfigs.config import ExperimentConfig
from src.ConstantsConfigs.constants import DEFAULT_PROJECT_PATH
from src.DataPrep.datamodule import TextClassificationDatamodule
from src.NN.nn_model import BERTModelClassic

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')


def train(cfg: ExperimentConfig) -> None:
    """
    Train loop

    :param cfg: file with exp config
    :return:
    """
    pl.seed_everything(0)
    datamodule = TextClassificationDatamodule(cfg=cfg.data_config)
    callbacks = [
        ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max', every_n_epochs=5),
    ]

    model = BERTModelClassic(cfg=cfg.module_config)
    trainer = pl.Trainer(callbacks=callbacks, **dict(cfg.trainer_config))
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    cfg_model_path_yaml = os.getenv(
        'TRAIN_CFG_PATH',
        DEFAULT_PROJECT_PATH / 'configs' / 'train.yaml',
    )
    train(cfg=ExperimentConfig.from_yaml(cfg_model_path_yaml))
