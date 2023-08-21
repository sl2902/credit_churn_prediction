import os
import sys
sys.path.append("scripts")
import pickle

import mlflow
import pandas as pd
import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient

import settings
import preprocess_data

# from mlflow.tracking import MlflowClient


MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://mongo:27017/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
if MLFLOW_TRACKING_URI == "":
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    print(f"Tracking URI {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Production')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'credit-churn-experiment')
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "False") == "True"
PICKLE_PATH = os.getenv("PICKLE_PATH", "pickle")
EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "127.0.0.1:8085")
TEST = os.getenv("TEST", "True") == "True"
MODEL_LOCATION = os.getenv('MODEL_LOCATION', "./model/")
print(f'model location {MODEL_LOCATION}')

mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")

app = Flask('credit-card-churn-app')


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def model_uri(stage, model_name):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_ENABLED:
        print("S3 model used")
        return f"models:/{model_name}/{stage}"

    if MODEL_LOCATION is not None:
        print('Local model used')
        return MODEL_LOCATION


def load_model(stage, model_name):
    uri = model_uri(stage, model_name)
    try:
        return mlflow.sklearn.load_model(model_uri=uri)
    except:
        print('S3 model failed')
        return mlflow.sklearn.load_model(model_uri=MODEL_LOCATION)


try:
    dv = load_pickle(f'{PICKLE_PATH}/dv.pkl')
except FileNotFoundError:
    dv = load_pickle('dv.pkl')
try:
    oe = load_pickle(f'{PICKLE_PATH}/ohe.pkl')
except FileNotFoundError:
    oe = load_pickle('ohe.pkl')


model = load_model(MODEL_STAGE, MLFLOW_MODEL_NAME)


@app.route('/predict', methods=["POST"])
def make_prediction():
    payload = request.get_json()
    payload, _, _ = preprocess_data.data_prep(
        pd.DataFrame(payload, index=[0]),
        oe,
        dv,
        is_train=False,
        is_drop=False,
        is_ohe=False,
        is_pred=True,
    )
    pred = model.predict_proba(payload)[:, 1]
    result = {
        'is_churned': "Yes" if pred[0] > 0.5 else "No",
        'model_version': f"{MLFLOW_MODEL_NAME}/{MODEL_STAGE}",
    }
    if not TEST:
        save_to_db(payload, pred[0])
        send_to_evidently_service(payload, pred[0])

    return jsonify(result)


def save_to_db(payload, prediction):
    row = payload.to_dict(orient="records")
    row[0]['prediction'] = "Yes" if prediction > 0.5 else "No"
    collection.insert_one(row[0])


def send_to_evidently_service(payload, prediction):
    payload = payload[settings.monitor].copy()
    row = payload.to_dict(orient="records")
    row[0]['prediction'] = 1 if prediction > 0.5 else 0
    # print(f'Evidently {row}')
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/churn", json=row)


if __name__ == "__main__":
    PORT = os.getenv("PORT", "9696")
    app.run(debug=True, host='0.0.0.0', port=int(PORT))
