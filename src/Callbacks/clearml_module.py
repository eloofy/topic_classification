from typing import Dict, Optional

from clearml import OutputModel, Task
from pytorch_lightning import Callback, LightningModule, Trainer

from src.ConstantsConfigs.config import ExperimentConfig


class ClearMLTracking(Callback):
    def __init__(
        self,
        cfg: ExperimentConfig,
        label_enumeration: Optional[Dict[str, int]] = None,
    ):
        """
        Constructor for clearml tracking callback

        :param cfg: cfg of experiment
        :param label_enumeration: labels
        """
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Create clearml before start

        :param trainer: trainer of model
        :param pl_module: main model
        :return:
        """
        self._setup_task()

    def _setup_task(self) -> None:
        """
        Setup clearml
        :return:
        """
        Task.force_requirements_env_freeze()
        self.task = Task.init(
            project_name=self.cfg.project_name,
            task_name=self.cfg.experiment_name,
            output_uri=True,
        )
        self.task.connect_configuration(configuration=self.cfg.model_dump())
        self.output_model = OutputModel(task=self.task, label_enumeration=self.label_enumeration)
