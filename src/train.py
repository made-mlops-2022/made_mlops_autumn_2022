import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import logging
import hydra
from omegaconf import DictConfig


@dataclass
class Config:
    config_name: str
    path_to_csv_train: str
    path_to_csv_test_X: str
    path_to_csv_test_y: str
    path_to_save_model: str
    path_to_write_metrics: str
    model: str


def print_metrics(model, config):
    with open(config.path_to_csv_test_X) as file_csv:
        X_test = pd.read_csv(file_csv)
    
    with open(config.path_to_csv_test_y) as file_csv:
        y_test = pd.read_csv(file_csv)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics_text = f'{accuracy=}\n{f1=}'
    logging.info(f'{accuracy=}')
    logging.info(f'{f1=}')

    with open(config.path_to_write_metrics, 'w') as f:
        f.write(metrics_text)


def train(config) -> None:    
    with open(config.path_to_csv_train) as file_csv:
        X = pd.read_csv(file_csv)
    
    y = X['condition']
    X = X.drop('condition', axis=1)

    logging.info('train model...')
    
    if config.model == 'LogisticRegression':
        model = LogisticRegression()
    if config.model == 'RandomForest':
        model = RandomForestClassifier()
    
    model.fit(X, y)

    dump(model, config.path_to_save_model)
    logging.info(f'model was saved to {config.path_to_save_model}...')

    print_metrics(model, config) 


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_train(cfg: DictConfig) -> None:
    config_dict = cfg.train
    config = Config(**config_dict)
    train(config)


if __name__ == '__main__':
    logging.basicConfig(
        filename="../log/train.log",
        level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )

    run_train()
