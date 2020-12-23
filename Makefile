format:
	black . --exclude checkpoint --exclude wandb --exclude save
	isort . --skip checkpoint --skip wandb --skip save

test:
	black . --check --exclude checkpoint --exclude wandb --exclude save
	isort . --check-only --skip checkpoint --skip wandb --skip save
	env PYTHONPATH=. pytest --pylint --flake8 --mypy --ignore=checkpoint --ignore=wandb --ignore=save --ignore=config

install:
	conda env create -f environment.yml 

dev:
	pip install pre-commit
	pre-commit install

docker-push:
	docker build -t jmarpledev/model_compression .
	docker push jmarpledev/model_compression
