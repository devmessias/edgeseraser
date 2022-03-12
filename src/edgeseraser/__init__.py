"""This pkg aims to implement serveral filtering methods for (un)directed graphs.

Edge filtering methods allows to extract the backbone of a graph
or sampling the most important ones. You can use these methods
as a preprocessing step aiming to improve the performance/results of
graph algorithms or to turn a graph visualtzation more asthetic.

**See the example below for a simple usage of the package.**
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

"""

__author__ = """Bruno Messias"""
__email__ = "devmessias@gmail.com"
__version__ = "0.1.0"
