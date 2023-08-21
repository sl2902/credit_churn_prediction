FROM python:3.8.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv lock

RUN pipenv install --system --deploy

# COPY [ "pickle/*.pkl", "services/predict.py", "scripts/train.py", "scripts/preprocess_data.py", "scripts/settings.py", "./" ]
COPY [ "pickle/*.pkl", "services/*.py", "scripts/*.py", "scripts/blocks/*.py", "./" ]

# EXPOSE 9696 9696

# ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]