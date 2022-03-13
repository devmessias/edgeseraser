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

## What is Edges Eraser?
This pkg aims to implement serveral filtering methods for (un)directed graphs.

Edge filtering methods allows to extract the backbone of a graph or sampling the most important edges. You can use edge filtering methods as a preprocessing step aiming to improve the performance/results of graph algorithms or to turn a graph visualtzation more asthetic.


## Example
```python
import networkx as nx
import edgeseraser as ee

g = nx.erdos_renyi_graph(100, 0.1)
ee.noise_score.filter_nx_graph(g, field=None)

g # filtered graph
```

## Available methods and details

| Method | Description | suitable for | limitations/restrictions/details |
| --- | --- |--- | --- |
| [Noise Score] | Filters edges with high noise score. Paper:[1]|Directed, Undirected, Weighted | Very good and fast! [4] |
| [Disparity] | Dirichlet process filter (stick-breaking) Paper:[2] |  Directed, Undirected, Weighted |There are some criticism regarding the use in undirected graphs[3]|

[1]: https://arxiv.org/abs/1701.07336
[2]: https://arxiv.org/abs/0904.
[3]: https://arxiv.org/abs/2101.00863
[4]: https://www.michelecoscia.com/?p=1236
[Noise Score]: /api_docs/#edgeseraser.noise_score
[Disparity]: /api_docs/#edgeseraser.disparity


## Installation

```sh
pip install edgeseraser
```

## Development

* Clone/Fork this repository

```sh
git clone https://github.com/devmessias/edgeseraser
```

* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+

```sh
make install
make init
```

### Testing

```sh
make test
```

To run the static analysis, use the following command:
```sh
make mypy
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

To see the current state of the documentation in your browser, use the following command:
```sh
make docs-serve
```
The above command will start a local server on port 8000. Any changes to
the documentation and docstrings will be automatically reflected in your browser.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.


If you want e.g. want to run all checks manually for all files:

```sh
make pre-commit
```
