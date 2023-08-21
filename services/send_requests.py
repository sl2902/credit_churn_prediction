import os
import sys
import pickle
import logging
import argparse
from time import sleep

import pandas as pd
import requests

sys.path.append("scripts")
import settings
import preprocess_data

FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
PORT = os.getenv("PORT", "9696")
url = f"http://localhost:{PORT}/predict"
print(url)


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
    # logging.info("Read test file")
    df = pd.read_csv(file_path, sep=",", encoding="latin1")

    new_df = df[settings.cols_to_keep].copy()
    preprocess_data.clean_names(new_df)
    test = preprocess_data.split_dataset(new_df)[2]

    # payload = {"customer_age": 48, "dependent_count": 3, "gender": "F", "education_level": "Uneducated",
    #             "marital_status": "Single", "income_category": "Less than $40K", "card_category": "Blue",
    #             "months_on_book": 39, "total_relationship_count": 4, "credit_limit": 2991.0,
    #               "total_revolving_bal": 1508}
    # resp = requests.post(url, json=payload).json()
    # print(f"Has customer churned?: {resp['is_churned']}")

    # print(resp)
    for row in test.itertuples():
        payload = {
            "customer_age": row.customer_age,
            "dependent_count": row.dependent_count,
            "gender": row.gender,
            "education_level": row.education_level,
            "marital_status": row.marital_status,
            "income_category": row.income_category,
            "card_category": row.card_category,
            "months_on_book": row.months_on_book,
            "total_relationship_count": row.total_relationship_count,
            "credit_limit": row.credit_limit,
            "total_revolving_bal": row.total_revolving_bal,
        }
        resp = requests.post(
            url,
            # # headers={"Content-Type": "application/json"},
            json=payload,
        ).json()
        print(f"Has customer churned? {resp['is_churned']}")
        sleep(0.5)


if __name__ == "__main__":
    main()
