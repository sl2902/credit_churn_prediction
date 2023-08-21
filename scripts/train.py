import os
import sys
import pickle
import logging
import argparse
from pathlib import Path

# from sklearn.preprocessing import OrdinalEncoder
# from sklearn.feature_extraction import DictVectorizer
import mlflow

# from dotenv import load_dotenv
import optuna
import pandas as pd

# import settings
import preprocess_data
from prefect import flow, task
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from prefect.task_runners import SequentialTaskRunner
from sklearn.linear_model import LogisticRegression

# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score

# load_dotenv(".env")


FORMAT = "%(asctime)s %(filename)s %(levelname)s %(message)s"

logging.basicConfig(format=FORMAT, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)
PICKLE_PATH = os.getenv("PICKLE_PATH", "pickle")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "credit-churn-experiment")

if MLFLOW_TRACKING_URI == "":
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    print(f"Tracking URI {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


@task
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


@task()
def train_model(train_X, train_y, valid_X, valid_y, test_X, test_y, cv=3, num_trials=3):
    """
    Train and optimize model
    """
    # np.random.seed(42)
    # start_time = time.time()
    mlflow.sklearn.autolog()

    def objective(trial):
        params = {
            'solver': 'liblinear',
            'class_weight': 'balanced',
            "C": trial.suggest_float('C', 0.01, 0.1, log=True),
            'random_state': 42,
            # 'n_jobs': -1
        }
        with mlflow.start_run():
            mlflow.set_tag("model", "LogisticRegression")
            mlflow.log_params(params)

            lr = LogisticRegression(**params)
            lr.fit(train_X, train_y)
            y_pred = lr.predict_proba(valid_X)[:, 1]
            roc_auc = roc_auc_score(valid_y, y_pred)
            # print(f"ROC is {roc_auc}")
            mlflow.log_metric("roc_auc", roc_auc)
            # throws the following error inside docker container
            # An error occurred (InvalidAccessKeyId) when calling the PutObject operation:
            # The AWS Access Key Id
            # you provided does not exist in our records.
            # mlflow.sklearn.log_model(lr, artifact_path="churn_predictor")
            skfold = StratifiedKFold(n_splits=cv)
            scores = cross_val_score(
                lr, valid_X, valid_y, cv=skfold, scoring='roc_auc_ovr_weighted'
            )
            # print(f"weighted roc auc cross val score {scores.mean()}")

        return scores.mean()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)
    logger.info(
        f"Best score {study.best_value}. Best params {study.best_params}. Best trial {study.best_trial}"
    )
    # best_trial = study.best_trial
    # params = study.best_params
    # best_lr = LogisticRegression(solver='liblinear', class_weight='balanced', **params)
    # best_lr.fit(train_X,train_y)
    # y_pred = best_lr.predict_proba(test_X)[:, 1]
    # best_roc_auc = roc_auc_score(test_y, y_pred)
    # mlflow.log_metric("roc_auc", best_roc_auc)
    # mlflow.sklearn.log_model(best_lr, artifact_path="churn_predictor")


@task
def register_model(metric='roc_auc', registered_model_name=EXPERIMENT_NAME):
    client = MlflowClient()
    # select the prediction_service with the lowest test log loss
    experiment = client.get_experiment_by_name(registered_model_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )[0]
    # logger.info(f"Test roc auc {best_run.data.metrics[metric]}")
    print(f"Test roc auc {best_run.data.metrics[metric]}")
    # Register the best prediction_service
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri, name=registered_model_name
    )
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )


@flow(log_prints=True, task_runner=SequentialTaskRunner())
def main():
    parser = argparse.ArgumentParser(description="Train best model")
    parser.add_argument(
        "--file-path",
        "-f",
        help="Enter file path and name of file to train",
        default="data/BankChurnersSample.csv",
    )
    args = parser.parse_args()
    file_path = args.file_path

    print(f"Tracking URI {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logging.info("Read input file")
    df = pd.read_csv(file_path, sep=",", encoding="latin1")
    preprocess_data.preprocess_pipeline(df)

    Path(f"{PICKLE_PATH}").mkdir(parents=True, exist_ok=True)
    try:
        train_x, train_y = load_pickle(f'{PICKLE_PATH}/train.pkl')
    except:
        train_x, train_y = load_pickle('train.pkl')
    try:
        val_x, val_y = load_pickle(f'{PICKLE_PATH}/val.pkl')
    except:
        val_x, val_y = load_pickle('val.pkl')
    try:
        test_x, test_y = load_pickle(f'{PICKLE_PATH}/test.pkl')
    except:
        test_x, test_y = load_pickle('test.pkl')

    logger.info("Train the model")
    train_model(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        cv=3,
        num_trials=10,
        # scoring=make_scorer(roc_auc_scorer, needs_threshold=True),
    )
    logger.info("Register model")
    register_model()
    logger.info("Registration completed")

    # logging.info("Pickle the model using Bentoml")
    # bentoml.sklearn.save_model(
    #     'credit_card_churn_model',
    #     best_lr_model,
    #     custom_objects=
    #     {
    #         "dictVectorizer": dv
    #     },
    #     signatures={
    #       "predict_proba": {
    #                     "batchable": True,
    #                     "batch_dim": 0
    #                     }
    #     }
    # )


if __name__ == "__main__":
    main()
