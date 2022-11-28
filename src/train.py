import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import logging
import hydra
from omegaconf import DictConfig


@dataclass
class Config:
    config_name: str
    path_to_csv: str
    path_to_save_model: str
    model: str


def train(config) -> None:    
    with open(config.path_to_csv) as file_csv:
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
