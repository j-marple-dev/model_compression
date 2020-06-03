format:
	black . --exclude checkpoint --exclude wandb
	isort -y --skip checkpoint --skip wandb

test:
	black . --check --exclude checkpoint --exclude wandb
	isort -y --check-only --skip checkpoint --skip wandb
	env PYTHONPATH=. pytest --pylint --flake8 --ignore=checkpoint --ignore=wandb

dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

dep:
	pip install -r requirements.txt

docker-push:
	docker build -t jmarpledev/model_compression .
	docker push jmarpledev/model_compression
