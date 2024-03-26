from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class _BaseValidatedConfig(BaseModel):
    """
    Validated config with extra='forbid'
    """

    model_config = ConfigDict(extra='forbid')


class SerializableOBj(_BaseValidatedConfig):
    """
    SerializableOBj cfg for import
    """

    target_class: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class DataConfig(_BaseValidatedConfig):
    """
    Data config
    """

    batch_size: int = 64
    train_size: float = 0.8
    pin_memory: bool = True
    num_samples: int = 116243
    shuffle: bool = True
    dataset_name: str = 'data_topic_soc_dem_full.xlsx'
    task_name: str = 'social_dem'
    pretrained_tokenizer: str = 'data_topic_soc_dem_full_end2end.xlsx'


class ModelConfig(_BaseValidatedConfig):
    name_model: str = 'BERT'
    pretrained: bool = False
    pretrained_model: str = 'MonoHime/rubert-base-cased-sentiment-new'
    num_classes: int = 29
    optimizer: SerializableOBj = SerializableOBj(
        target_class='torch.optim.AdamW',
        kwargs={'lr': 1e-4, 'weight_decay': 1e-1},
    )
    vocab_size: int = 100792
    hidden_size: int = 768
    embed_size: int = 900
    num_layers: int = 6
    num_attention_heads: int = 32


class TrainerConfig(_BaseValidatedConfig):
    """
    Trainer config
    """

    min_epochs: int = 20
    max_epochs: int = 30
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: Optional[float] = 0.1
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = 'norm'
    deterministic: bool = False
    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None
    detect_anomaly: bool = False
    accelerator: str = 'gpu'
    devices: List = [0]
    logger: bool = True


class ExperimentConfig(_BaseValidatedConfig):
    """
    Experiment config
    """

    project_name: str = 'BERTClassification'
    experiment_name: str = 'exp_main_base_bert'
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())
    module_config: ModelConfig = Field(default=ModelConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load config from yaml
        :param path: path to model config yaml
        :return: config container from yaml
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save config
        :param path: yaml path
        :return:
        """
        with open(path, 'w') as out_file:
            yaml.safe_dump(
                self.model_dump(),
                out_file,
                default_flow_style=False,
                sort_keys=False,
            )
