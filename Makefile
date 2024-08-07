SOURCE_DIR=.
IMAGE_NAME=lensor-api

install:
	poetry install

run:
	poetry run python -m src.api.main

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -p 8087:8087 -p 6006:6006 --env-file .env $(IMAGE_NAME)

isort:
	@echo "> Running isort.." && poetry run isort $(SOURCE_DIR)

black:
	@echo "> Running black.." && poetry run black $(SOURCE_DIR)

pylint:
	@echo "> Running pylint.." && poetry run pylint $(SOURCE_DIR)

autoflake:
	@echo "> Running autoflake.." && poetry run autoflake -r $(SOURCE_DIR)

isort-check:
	@echo "> Running isort (check).." && poetry run isort $(SOURCE_DIR) --check 

black-check:
	@echo "> Running black (check).." && poetry run black $(SOURCE_DIR) --check

autoflake-check:
	@echo "> Running autoflake.." && poetry run autoflake -r -c $(SOURCE_DIR)

fmt: isort black autoflake

lint: isort-check black-check autoflake-check