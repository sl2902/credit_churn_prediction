import os
import sys
import pickle
import logging
import argparse
from pathlib import Path

import pandas as pd
import settings

# import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, make_scorer
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.ensemble import RandomForestClassifier


FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)
PICKLE_PATH = os.getenv("PICKLE_PATH", "pickle")


def dump_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def clean_names(df):
    """Rename some of the columns in the DataFrame"""
    df.rename(columns=settings.col_rename, inplace=True)


def split_dataset(df, size1=0.2, size2=0.25, random_state=42):
    """Split input DataFrame into train, validation and test"""
    train_df, test_x = train_test_split(df, test_size=size1, random_state=random_state)
    train_x, val_x = train_test_split(
        train_df, test_size=size2, random_state=random_state
    )
    return (
        train_x.reset_index(drop=True),
        val_x.reset_index(drop=True),
        test_x.reset_index(drop=True),
    )


def transform_data(df, oe, dv, is_ohe=False, is_train=False):
    if not is_ohe:
        cat_fields = settings.cat_fields
        if is_train:
            df[cat_fields] = oe.fit_transform(df[cat_fields])
        else:
            df[cat_fields] = oe.transform(df[cat_fields])
    dicts = df.to_dict(orient='records')
    if is_train:
        dv.fit(dicts)
    df = dv.transform(dicts)

    return df


def data_prep(
    df, oe, dv, is_ohe=False, is_train=True, is_test=False, is_pred=False, is_drop=True
):
    """Prepare dataset"""
    # df = split_dataset(df)[split]

    if not is_pred:
        y = df["attrition_flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)

    try:
        df = df.drop(["client_num", "attrition_flag"], axis=1)
    except:
        pass

    if is_drop:
        try:
            for col in [
                "months_on_book",
                "total_relationship_count",
                "total_revolving_bal",
            ]:
                del df[col]
        except:
            pass

    # if not is_ohe:
    df = transform_data(df, oe, dv, is_ohe=is_ohe, is_train=is_train)
    # cat_fields = settings.cat_fields
    # if is_train:
    #     df[cat_fields] = oe.fit_transform(df[cat_fields])
    # else:
    #     df[cat_fields] = oe.transform(df[cat_fields])
    # dicts = df.to_dict(orient='records')
    # if is_train:
    #     dv.fit(dicts)
    # df = dv.transform(dicts)
    #     if is_train:
    #         scaler.fit(df)
    #     df = scaler.transform(df)
    df = pd.DataFrame(df, columns=dv.get_feature_names_out())

    if not is_pred:
        return df, y, dv, oe
    return df, dv, oe


def preprocess_pipeline(df):
    # take a subset of the columns
    new_df = df[settings.cols_to_keep].copy()
    logging.info("Clean the field names")
    clean_names(new_df)
    logging.info("Split and prepare the train, validation and test sets")
    train = split_dataset(new_df)[0]
    train_x, train_y, dv, ohe = data_prep(
        train, settings.oe, settings.dv, is_drop=False, is_pred=False, is_ohe=False
    )
    val = split_dataset(new_df)[1]
    val_x, val_y, _, _ = data_prep(
        val,
        settings.oe,
        settings.dv,
        is_train=False,
        is_drop=False,
        is_pred=False,
        is_ohe=False,
    )
    test = split_dataset(new_df)[2]
    test_x, test_y, _, _ = data_prep(
        test,
        settings.oe,
        settings.dv,
        is_train=False,
        is_test=False,
        is_drop=False,
        is_pred=False,
        is_ohe=False,
    )
    logging.info("Pickle dict_vectorizer and one_hot_encoder objects")
    Path(f"{PICKLE_PATH}").mkdir(parents=True, exist_ok=True)
    try:
        dump_pickle(dv, f"{PICKLE_PATH}/dv.pkl")
    except:
        dump_pickle(dv, "dv.pkl")
    try:
        dump_pickle(ohe, f"{PICKLE_PATH}/ohe.pkl")
    except:
        dump_pickle(ohe, "ohe.pkl")

    logging.info("Pickle the train, test and validation datasets")
    try:
        dump_pickle((train_x, train_y), f"{PICKLE_PATH}/train.pkl")
    except:
        dump_pickle((train_x, train_y), "train.pkl")
    try:
        dump_pickle((val_x, val_y), f"{PICKLE_PATH}/val.pkl")
    except:
        dump_pickle((val_x, val_y), "val.pkl")
    try:
        dump_pickle((test_x, test_y), f"{PICKLE_PATH}/test.pkl")
    except:
        dump_pickle((test_x, test_y), "test.pkl")
    logging.info(f"Train size {train_x.shape}")
    logging.info(f"Validation size {val_x.shape}")
    logging.info(f"Test size {test_x.shape}")


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
    preprocess_pipeline(df)


if __name__ == "__main__":
    main()
