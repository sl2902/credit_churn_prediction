import os
import sys
import pickle
import logging

# import time
import argparse

import pandas as pd

# import settings
import preprocess_data
from dotenv import load_dotenv

# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.feature_extraction import DictVectorizer


load_dotenv()


FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"

logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
PICKLE_PATH = os.getenv("PICKLE_PATH", "pickle")


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train best model")
    parser.add_argument(
        "--file-path",
        "-f",
        required=True,
        help="Enter file path and name of file to train",
    )
    args = parser.parse_args()
    file_path = args.file_path

    logging.info("Read input file")
    df = pd.read_csv(file_path, sep=",", encoding="latin1")
    df = df.sample(n=5000, random_state=42)
    preprocess_data.clean_names(df)
    logging.info(f'Shape of dataset {df.shape}')

    try:
        dv = load_pickle(f'{PICKLE_PATH}/dv.pkl')
    except FileNotFoundError:
        dv = load_pickle('dv.pkl')
    try:
        oe = load_pickle(f'{PICKLE_PATH}/ohe.pkl')
    except FileNotFoundError:
        oe = load_pickle('ohe.pkl')

    df = preprocess_data.transform_data(df, oe, dv, is_ohe=False, is_train=False)
    df = pd.DataFrame(df, columns=dv.get_feature_names_out())
    df.to_csv("ref_data/reference_data.csv", index=False)


if __name__ == "__main__":
    main()
