from typing import Any, List, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix
from torchinfo import summary
from sklearn.metrics import f1_score

from src.Callbacks.clearml_module import ClearMLTracking
from src.ConstantsConfigs.constants import DECODE_TOPIC


class LogModelSummary(Callback):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Callback to save model summary prev start

        :param trainer: trainer of model
        :param pl_module: main model
        :return:
        """
        text = next(iter(trainer.train_dataloader))['input_ids']

        text = text.to(pl_module.device)
        summary(pl_module.model, input_data=text)


class PredictsCallbackBase(Callback):
    def __init__(self, clearml_task: ClearMLTracking, every_n_epoch: int, labels: List[str]):
        """
        Constructor for Confusion Matrix Callback

        :param clearml_task: created task clearml
        :param every_n_epoch: save each n epoch
        """
        super().__init__()
        self.clearml_task = clearml_task
        self.every_n_epoch = every_n_epoch
        self.labels = labels
        self.predicts: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Clear saved predicts and labels

        :param trainer: trainer of model
        :param pl_module: main model
        :return:
        """
        self._reset()

    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Save predicts from batch data

        :param trainer: trainer of model
        :param pl_module: main model
        :param outputs: predicts val
        :param batch: batch data
        :param batch_idx: batch data idx
        :param dataloader_idx: dataloader idx
        :return:
        """
        self._store_outputs(batch, outputs)

    def _reset(self):
        """
        Clear update predicts
        :return:
        """
        self.predicts = []
        self.targets = []

    def _store_outputs(self, batch: Dict[str, torch.Tensor], outputs: torch.Tensor) -> None:
        """
        Adв batch outputs and labels

        :param batch: batch
        :param outputs: outputs predicts
        :return:
        """
        self.predicts.append(outputs)
        self.targets.append(batch['label'])


class ConfusionMatrix(PredictsCallbackBase):
    def __init__(self, clearml_task: ClearMLTracking, every_n_epoch: int, labels: List[str]):
        super().__init__(clearml_task, every_n_epoch, labels)

    def on_validation_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ) -> None:
        """
        Check epoch and log confusion matrix

        :param trainer: trainer of model
        :param pl_module: main model
        :return:
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            self._log_confusion_matrix(trainer)

    def _log_confusion_matrix(self, trainer: 'pl.Trainer'):
        """
        Log confusion matrix

        :param trainer: trainer of model
        :return:
        """
        targets = (
            torch.cat(
                self.targets,
                dim=0,
            )
            .detach()  # noqa: WPS348
            .cpu()  # noqa: WPS348
            .numpy()  # noqa: WPS348
        )
        predicts = (
            torch.cat(
                self.predicts,
                dim=0,
            )
            .detach()  # noqa: WPS348
            .cpu()  # noqa: WPS348
            .numpy()  # noqa: WPS348
        )
        cf_matrix = confusion_matrix(targets, predicts, normalize='true')
        self.clearml_task.task.logger.current_logger().report_confusion_matrix(
            f'Confusion matrix: epoch {trainer.current_epoch}',
            'ignored',
            xaxis='Predicted',
            yaxis='Actual',
            xlabels=self.labels,
            ylabels=self.labels,
            matrix=cf_matrix,
        )


class EachClassPercentCallback(PredictsCallbackBase):
    def __init__(self, clearml_task: ClearMLTracking, every_n_epoch: int, labels: List[str]):
        super().__init__(clearml_task, every_n_epoch, labels)

    def on_validation_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ) -> None:
        """
        Check epoch and log confusion matrix

        :param trainer: trainer of model
        :param pl_module: main model
        :return:
        """
        if trainer.current_epoch % self.every_n_epoch == 0:
            self._log_each_class_metric(trainer)

    def _log_each_class_metric(self, trainer):
        targets = (
            torch.cat(
                self.targets,
                dim=0,
            )
            .detach()  # noqa: WPS348
            .cpu()  # noqa: WPS348
            .numpy()  # noqa: WPS348
        )
        predicts = (
            torch.cat(
                self.predicts,
                dim=0,
            )
            .detach()  # noqa: WPS348
            .cpu()  # noqa: WPS348
            .numpy()  # noqa: WPS348
        )

        f1_scores = f1_score(targets, predicts, average=None)

        for label, scalar in zip(list(DECODE_TOPIC['social_dem'].keys()), f1_scores):
            self.clearml_task.task.logger.current_logger().report_scalar(
                'Each Class f1',
                label,
                iteration=trainer.current_epoch,
                value=scalar
            )
