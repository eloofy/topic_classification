import os
from pathlib import Path

from src.ConstantsConfigs.config import ExperimentConfig
from src.ConstantsConfigs.constants import DECODE_TOPIC, DEFAULT_PROJECT_PATH
from src.Inference.streamlit_class import TextClassifierApp


def run_stream():
    cfg = ExperimentConfig()
    best_path = Path(
        'src/BestResults/BERT-epoch=4--mean_valid_loss=1.1646--valid_f1=0.6375.ckpt',
    )
    app = TextClassifierApp(
        Path(os.path.join(DEFAULT_PROJECT_PATH, best_path)),
        cfg,
        DECODE_TOPIC['social_dem'],
    )
    app.run()


if __name__ == '__main__':
    run_stream()
