.PHONY: clean-dist
ifeq (,$(shell which poetry))
$(echo "poetry not found. Please install it first. pip install poetry")
endif

init:
	poetry shell

install:
	poetry shell
	poetry update
	poetry install

install-dev: install
	pre-commit install

# update your dependencies
update:
	poetry shell
	poetry update
	poetry install

# Anything related to how health our codebase is
test:
	poetry run pytest

mypy:
	poetry run mypy edgeseraser

mypy-strict:
	mypy edgeseraser --strict | grep "^edgeseraser/"

pre-commit:
	pre-commit run --all-files

# build the pkgs inside dist
build:
	poetry build

# Documentation
docs-serve:
	poetry run mkdocs serve

docs-deploy:
	poetry run mkdocs gh-deploy --force


# Helps to install as pip pkg if you are using other envs
clean-dist:
	find -L dist/ -name '*.whl' -type f -delete

uninstall-pkg:
	pip uninstall edgeseraser -y

install-pip: uninstall-pkg
	$(eval FILE_WHL := $(shell find -L dist/ -name '*.whl' | sort | tail -n 1))
	pip install $(FILE_WHL)

install-pkg: clean-dist build install-pip
