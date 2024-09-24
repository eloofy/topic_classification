import pandas as pd
from transformers import AutoTokenizer
import torch
from src.NN.nn_model import BERTModelClassic
from src.ConstantsConfigs.config import ExperimentConfig
from src.ConstantsConfigs.constants import DECODE_TOPIC_CAR_PREDICT
import tqdm

config = ExperimentConfig()
model = BERTModelClassic(
    config.module_config
)
model_path = '/files/private_data/topic_classification/src/lightning_logs/version_28/checkpoints/BERT_CAR-epoch=9--mean_valid_loss=0.4623--valid_f1=0.7323.ckpt'
device = torch.device('cuda:6')
model.load_state_dict(torch.load(
    model_path, map_location=device
)['state_dict'])

tokenizer = AutoTokenizer.from_pretrained(
    config.data_config.pretrained_tokenizer
)

data = pd.read_excel(
    'Посты.xlsx', index_col=0
)

texts = data['post text']
model.eval().to(device)
results = []
for text in tqdm.tqdm(texts):
    try:
        inputs = tokenizer.encode_plus(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                add_special_tokens=True
            )
        inputs.to(device)
        with torch.no_grad():
            output = model.predict_step(inputs, 0, 0)
            prediction = torch.argmax(output).item()
            results.append(DECODE_TOPIC_CAR_PREDICT[str(prediction)])
    except Exception:
        results.append('None')


data['post topic car'] = results

data.to_excel('result.xlsx')
