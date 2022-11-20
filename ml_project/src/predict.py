import pandas as pd
from joblib import load
import argparse
import os.path
import logging


def predict(path_to_csv='../data/test.csv', path_to_model="../models/log_regression.joblib", path_to_write_csv="../data/predictions.csv"):
    with open(path_to_csv) as file_csv:
        X_test = pd.read_csv(file_csv)

    logging.info('loading model...')
    model = load(path_to_model) 

    logging.info('predict...')
    y_pred = model.predict(X_test)

    df = pd.DataFrame(y_pred, columns=['condition'])
    df.to_csv(path_to_write_csv, index=False)
    logging.info(f'predictions was writen to {path_to_write_csv}...')


def check_args(args):
    if not os.path.exists(args.path_to_csv):
        text_error = f'file {args.path_to_csv} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)

    if not os.path.exists(args.path_to_model):
        text_error = f'file {args.path_to_model} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)

    path_to_write_csv = os.path.dirname(args.args.path_to_csv)

    if not os.path.exists(path_to_write_csv):
        text_error = f'directory {path_to_write_csv} dont exist'
        logging.error(text_error)
        raise FileNotFoundError(text_error)


if __name__ == '__main__':
    logging.basicConfig(
        filename="../log/predict.log",
        level=logging.DEBUG,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )

    logging.error('start predict')

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_csv", type=str, help="path to test csv")
    parser.add_argument("path_to_model", type=str, help="path to model")
    parser.add_argument("path_to_write_csv", type=str, help="path to write predictions")
    args = parser.parse_args()

    predict(args.path_to_csv, args.path_to_model, args.path_to_write_csv)
