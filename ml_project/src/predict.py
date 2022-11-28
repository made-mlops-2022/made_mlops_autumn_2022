import pandas as pd
from joblib import load
import argparse
import os.path
import logging
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig


@dataclass
class Config:
    config_name: str
    path_to_csv: str
    path_to_model: str
    path_to_write_csv: str


def predict(config) -> None:
    with open(config.path_to_csv) as file_csv:
        X_test = pd.read_csv(file_csv)

    logging.info('loading model...')
    model = load(config.path_to_model) 

    logging.info('predict...')
    y_pred = model.predict(X_test)

    df = pd.DataFrame(y_pred, columns=['condition'])
    df.to_csv(config.path_to_write_csv, index=False)
    logging.info(f'predictions was writen to {config.path_to_write_csv}...')



def check_config(config) -> None:
    if not os.path.exists(config.path_to_csv):
        text_error = f'file {config.path_to_csv} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)

    if not os.path.exists(config.path_to_model):
        text_error = f'file {config.path_to_model} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)

    path_to_write_csv = os.path.dirname(config.path_to_csv)

    if not os.path.exists(path_to_write_csv):
        text_error = f'directory {path_to_write_csv} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_predict(cfg: DictConfig) -> None:
    config_dict = cfg.predict
    config = Config(**config_dict)

    check_config(config)
    predict(config)


if __name__ == '__main__':
    logging.basicConfig(
        filename="../log/predict.log",
        level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )

    run_predict()