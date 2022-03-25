
.PHONY: clean-dist

#################################################################################
# GLOBALS                                                                       #
#################################################################################


#################################################################################
# BLOCK TO TEST PYTHON INSTALLATION                                             #
#################################################################################

# Test if python is installed

ifeq (,$(shell which poetry))
$(echo "poetry not found. Please install it first. pip install poetry")
endif


#################################################################################
# COMMANDS Install dependencies and the pkg                                                                    #
#################################################################################

## Install dependencies and the pkg
install:
	poetry shell
	poetry update
	poetry install
	pre-commit install


## update your dependencies
update:
	poetry shell
	poetry update
	poetry install

## Anything related to how health our codebase is
pytests:
	poetry run pytest --benchmark-skip

## Perform a static analysis of the codebase
mypy:
	poetry run mypy edgeseraser

## Perform a static analysis of the codebase using strict rules
mypy-strict:
	mypy edgeseraser --strict | grep "^edgeseraser/"

## Benchmark the codebase: make benchmark name="benchmark_name"
benchmark:
	pytest -svv tests --benchmark-autosave --benchmark-only --benchmark-save=$(name)

## Benchmark the codebase and save the results: make benchmark-compare csv="name_of_csv"
benchmark-compare:
	py.test-benchmark compare --csv=".benchmarks/${csv}"

## Run pre-commit hooks
pre-commit:
	pre-commit run --all-files

#################################################################################
# COMMANDS  Buid                                                                     #
#################################################################################


## build the pkgs inside dist folder
build:
	poetry build

## live-reload server that generates the docs
docs-serve:
	poetry run mkdocs serve

## deploy the docs to github pages (avoid use this prefer auto-deploy with actions)
docs-deploy:
	poetry run mkdocs gh-deploy --force


## Clean dist folder
clean-dist:
	find -L dist/ -name '*.whl' -type f -delete

## Uninstall the edgeseraser pkg
uninstall-pkg:
	pip uninstall edgeseraser -y

## Uninstall and install the edgeseraser pkg with pip
install-pip: uninstall-pkg
	$(eval FILE_WHL := $(shell find -L dist/ -name '*.whl' | sort | tail -n 1))
	pip install $(FILE_WHL)

## Clean, build and install via pip
install-pkg: clean-dist build install-pip



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
