from typing import Tuple

import networkx as nx
import numpy as np


def nx_extract(
    g, remap_labels: bool = False, field: str = None
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    nodes = g.nodes()
    num_vertices = len(nodes)
    try:
        nodes[0]
    except KeyError:
        remap_labels = True
    nodelabel2index = {}
    if remap_labels:
        nodelabel2index = {node: i for i, node in enumerate(nodes)}
        nx.relabel_nodes(g, nodelabel2index, copy=False)

    if field is None:
        edges = np.array([[u, v, 1.0] for u, v in g.edges()])
    else:
        edges = np.array([[u, v, d[field]] for u, v, d in g.edges(data=True)])
    weights = edges[:, 2].astype(np.float64)
    edges = edges[:, :2].astype(np.int64)

    return edges, weights, num_vertices, nodelabel2index


def nx_erase(g, edges2erase, nodelabel2index):
    g.remove_edges_from([(e[0], e[1]) for e in edges2erase])
    # check dict is empty
    if len(nodelabel2index) > 0:
        nx.relabel_nodes(
            g, {i: node for node, i in nodelabel2index.items()}, copy=False
        )


def ig_extract(
    g,
    field: str = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    num_vertices = g.vcount()

    edges = np.array(g.get_edgelist())
    if field is None:
        weights = np.ones(edges.shape[0])
    else:
        weights = np.array(g.es[field]).astype(np.float64)

    return edges, weights, num_vertices


def ig_erase(g, ids2erase):
    g.delete_edges(ids2erase)
