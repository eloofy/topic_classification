import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.ConstantsConfigs.config import ExperimentConfig
from src.ConstantsConfigs.constants import DEFAULT_PROJECT_PATH
from src.DataPrep.datamodule import TextClassificationDatamodule
from src.NN.nn_model import BERTModelClassic
from src.Callbacks.clearml_module import ClearMLTracking
from src.Callbacks.debug import ConfusionMatrix, LogModelSummary, EachClassPercentCallback
from src.ConstantsConfigs.constants import DECODE_TOPIC

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')


def train(cfg: ExperimentConfig) -> None:  # noqa: WPS210
    """
    Train loop

    :param cfg: file with exp config
    :return:
    """
    pl.seed_everything(0)
    datamodule = TextClassificationDatamodule(cfg=cfg.data_config)
    tracking_cb = ClearMLTracking(
        cfg, label_enumeration=DECODE_TOPIC[cfg.data_config.task_name],
    )
    confusion_tracking = ConfusionMatrix(
        tracking_cb,
        every_n_epoch=1,
        labels=list(DECODE_TOPIC[cfg.data_config.task_name].keys()))
    summary = LogModelSummary()
    each_class_f1 = EachClassPercentCallback(
        tracking_cb,
        every_n_epoch=1,
        labels=list(DECODE_TOPIC[cfg.data_config.task_name].keys())
    )
    callbacks = [
        tracking_cb,
        summary,
        confusion_tracking,
        each_class_f1,
        ModelCheckpoint(
            filename='BERT-{epoch}--{mean_valid_loss:.4f}--{valid_f1:.4f}',
            save_top_k=1,
            monitor='mean_valid_loss',
            mode='min',
            every_n_epochs=5,
        ),
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
