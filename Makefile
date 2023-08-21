SHELL:=/bin/bash

setup: ## Setup the development environment
	@pipenv install --dev; pipenv run pre-commit install; pipenv shell

build: ## Docker build and run the MLOps pipeline
	@docker-compose up --build -d

deployment: ## Deploy the scheduled training workflow
	@docker exec -t prefect python s3_remote_storage.py
	@docker exec -t prefect prefect deployment build ./train.py:main --name "Credit Churn Deployment" --tag credit-churn-experiment --cron "0 0 * * *" --storage-block remote-file-system/minio
	@docker exec -t prefect prefect deployment apply main-deployment.yaml
	@docker exec -td prefect prefect agent start --tag credit-churn-experiment

train: ## Execute the training workflow
	@docker exec -ti prefect prefect deployment run "main/Credit Churn Deployment"

logs: ## Check MLOps pipeline logs
	@docker-compose logs -f

post_request: ## Send requests to the server
	@docker exec -ti app python send_requests.py --file-path data/BankChurnersSample.csv

restart: ## Restart MLOps pipeline environment
	@docker-compose restart

stop: ## Stop MLOps pipeline environment
	@docker-compose down

clean: ## Clean environment
	# @docker container stop $(docker container ls -q)
	# @docker container prune
	# @docker rmi -f image $(docker image ls -q)
	@docker-compose down -v
	@docker-compose down --rmi=all

unit_tests: ## Run the unit tests
	@python -m pytest tests

quality_checks: ## Perform code quality checks
	isort scripts
	black scripts
	pylint --recursive=y scripts

run_docker_integration: ## Run docker-compose pipeline for integration tests
	@bash ./integration_tests/run.sh

run_integration_tests: ## Run integration tests
	@docker restart test_app
	@docker exec -t test_prefect python train.py
	@python integration_tests/test_predict.py
	@bash ./integration_tests/shutdown.sh

help: ## Help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
