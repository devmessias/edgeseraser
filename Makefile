ifeq (,$(shell which poetry))
$(error "poetry not found. Please install it first. pip install poetry")
endif

init:
	poetry shell

install:
	poetry shell
	poetry update
	poetry install
	pre-commit install

update:
	poetry shell
	poetry update
# Anything related to how health our codebase is
test:
	@poetry run pytest

mypy:
	@poetry run mypy .

pre-commit:
	pre-commit run --all-files


# Documentation
docs-serve:
	@poetry run mkdocs serve

docs-deploy:
	poetry run mkdocs gh-deploy --force
