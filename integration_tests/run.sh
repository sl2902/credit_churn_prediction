#!/usr/bin/env bash

# cd "$(dirname "$0")"
cd integration_tests

# function print_info {
#     RESET="\e[0m"
#     BOLD="\e[1m"
#     YELLOW="\e[33m"
#     echo -e "$YELLOW$BOLD [+] $1 $RESET"
# }

export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
export AWS_REGION=eu-west-1
export AWS_DEFAULT_REGION=eu-west-1
export AWS_ACCESS_KEY_ID=In1IKgTRbO9U4yPK
export AWS_SECRET_ACCESS_KEY=Ee2p53DTNVczXbZ62JLOBAzHAQ4cOnyY
export POSTGRES_DB=mlflowdb
export POSTGRES_USER=user
export POSTGRES_PASSWORD=password
export EXPERIMENT_NAME=test-credit-churn-experiment

echo "Creating MLOps test pipeline"
docker-compose up --build -d

if [[ $? -eq 0 ]]; then
    # sleep 5
    # echo "Wait a few seconds for pipeline to stabilise"
    echo "Pipeline ready..."
else
    unset EXPERIMENT_NAME
    docker-compose logs
    docker-compose down -v
    docker-compose down --rmi=all
    docker rmi -f image $(docker image ls -q)
    echo "Docker compose pipeline failed"
fi

# echo "Run prefect training workflow"
# docker exec -t test_prefect python train.py

# ERROR_CODE=$?

# if [[ $? -ne 0 ]]; then
#     unset EXPERIMENT_NAME
#     docker-compose logs
#     docker-compose down
#     docker-compose down -v
#     docker-compose down --rmi=all
#     docker rmi -f image $(docker image ls -q)
#     echo "Prefect training pipeline failed"
#     exit $?
# else
#     echo "Successful integration test of train.py"
# fi

# sleep 10

# echo "Reload model"
# docker restart test_app

# sleep 5

# echo "Test model serving"
# python test_predict.py

# ERROR_CODE=$?

# if [ ${ERROR_CODE} != 0 ]; then
#     unset EXPERIMENT_NAME
#     docker-compose logs
#     docker-compose down
#     docker-compose down -v
#     docker-compose down --rmi=all
#     docker rmi -f image $(docker image ls -q)
#     exit ${ERROR_CODE}
# else
#     echo "Successful integration test of predict.py"
# fi

# echo "Clean up"
# docker-compose down -v
# docker-compose down --rmi=all
# docker rmi -f image $(docker image ls -q)
# unset EXPERIMENT_NAME
# cd ../