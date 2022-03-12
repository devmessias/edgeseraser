# Edges Eraser

[![PyPI](https://img.shields.io/pypi/v/edgeseraser?style=flat-square)](https://pypi.python.org/pypi/edgeseraser/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/edgeseraser?style=flat-square)](https://pypi.python.org/pypi/edgeseraser/)
[![PyPI - License](https://img.shields.io/pypi/l/edgeseraser?style=flat-square)](https://pypi.python.org/pypi/edgeseraser/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://devmessias.github.io/edgeseraser](https://devmessias.github.io/edgeseraser)

**Source Code**: [https://github.com/devmessias/edgeseraser](https://github.com/devmessias/edgeseraser)

**PyPI**: [https://pypi.org/project/edgeseraser/](https://pypi.org/project/edgeseraser/)

---

A short description of the project

## Installation

```sh
pip install edgeseraser
```

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/devmessias/edgeseraser/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/devmessias/edgeseraser/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/devmessias/edgeseraser/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
