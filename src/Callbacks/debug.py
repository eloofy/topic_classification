from typing import Any, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix

from src.Callbacks.clearml import ClearMLTracking


class ConfusionMatrix(Callback):
    def __init__(self, each_epoch: int, labels: dict, clearml_task: ClearMLTracking):
        super().__init__()
        self.clearml_task = clearml_task
        self.each_epoch = each_epoch
        self.labels = labels
        self.predicts: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
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
        self._store_outputs(batch, outputs)

    def on_validation_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ) -> None:
        self._log_confusion_matrix(trainer)

    def _log_confusion_matrix(self, trainer: 'pl.Trainer'):
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
        cf_matrix = confusion_matrix(targets, predicts)
        self.clearml_task.task.logger.current_logger().report_confusion_matrix(
            f'Confusion matrix: epoch {trainer.current_epoch}',
            'ignored',
            xaxis='Predicted',
            yaxis='Actual',
            matrix=cf_matrix,
        )

    def _reset(self):
        self.predicts = []
        self.targets = []

    def _store_outputs(self, batch: List[torch.Tensor], outputs: torch.Tensor) -> None:
        self.predicts.append(outputs)
        self.targets.append(batch[1])
