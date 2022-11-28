import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import logging
import numpy as np
from sklearn import preprocessing
import hydra
from omegaconf import DictConfig


@dataclass
class Config:
    config_name: str
    path_to_csv: str
    dir_to_write: str
    test_size: float
    random_state: int
    normalizing: bool


def preprocessing_data(X, config) -> np.ndarray:
    if config.normalizing:
        X = preprocessing.normalize(X)
    return X


def split_csv(config) -> None:    
    with open(config.path_to_csv) as file_csv:
        df = pd.read_csv(file_csv)

    df = df.dropna()
    
    y = df['condition']
    X = df.drop('condition', axis=1)
    
    X = preprocessing_data(X, config)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, 
                                                        random_state=config.random_state, shuffle=False)

    columns = df.columns[:-1]
    X_train = pd.DataFrame(X_train, columns =[columns])
    X_train['condition'] = y_train
    X_test = pd.DataFrame(X_test, columns =[columns])
    
    X_train.to_csv(config.dir_to_write + 'train.csv', sep=',', encoding='utf-8', index=False)
    X_test.to_csv(config.dir_to_write + 'test.csv', sep=',', encoding='utf-8', index=False)
    y_test.to_csv(config.dir_to_write + 'y_true.csv', sep=',', encoding='utf-8', index=False)
    
    logging.info(f'train and test was writen to {config.dir_to_write}train.csv {config.dir_to_write}test.csv')


@hydra.main(version_base=None, config_path="../conf/", config_name="config")
def run_split_csv(cfg: DictConfig) -> None:
    config_dict = cfg.split
    config = Config(**config_dict)
    split_csv(config)


if __name__ == '__main__':
    logging.basicConfig(
        filename="../log/train_test_split.log",
        level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )

    logging.info('start train_test_split')

    run_split_csv()
