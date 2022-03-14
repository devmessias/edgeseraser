from typing import Optional, Tuple

import networkx as nx
import numpy as np


def nx_extract(
    g, remap_labels: bool = False, field: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    is_directed = g.is_directed()
    if is_directed:
        assert nx.is_weakly_connected(g), "Graph is not connected"
    else:
        assert nx.is_connected(g), "Graph is not connected"

    nodes = g.nodes()
    num_vertices = len(nodes)
    opts = {"is_directed": is_directed}
    try:
        nodes[0]
    except KeyError:
        remap_labels = True
    if remap_labels:
        nodelabel2index = {node: i for i, node in enumerate(nodes)}
        nx.relabel_nodes(g, nodelabel2index, copy=False)
        opts["nodelabel2index"] = nodelabel2index
    if field is None:
        edges = np.array([[u, v, 1.0] for u, v in g.edges()])
    else:
        edges = np.array([[u, v, d[field]] for u, v, d in g.edges(data=True)])
    assert edges.shape[0] > 0, "Graph is empty"
    weights = edges[:, 2].astype(np.float64)
    edges = edges[:, :2].astype(np.int64)
    return edges, weights, num_vertices, opts


def nx_erase(g, edges2erase, opts):
    g.remove_edges_from([(e[0], e[1]) for e in edges2erase])
    nodelabel2index = opts.get("nodelabel2index")
    if nodelabel2index:
        nx.relabel_nodes(
            g, {i: node for node, i in nodelabel2index.items()}, copy=False
        )


def ig_extract(
    g,
    field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int, dict]:
    is_directed = g.is_directed()
    if is_directed:
        components = g.components(mode="weak")
        assert len(components) == 1, "Graph is not connected"
    else:
        assert g.is_connected(), "Graph is not connected"

    num_vertices = g.vcount()
    opts = {"is_directed": is_directed}
    edges = np.array(g.get_edgelist())
    assert edges.shape[0] > 0, "Graph is empty"
    if field is None:
        weights = np.ones(edges.shape[0])
    else:
        weights = np.array(g.es[field]).astype(np.float64)
    return edges, weights, num_vertices, opts


def ig_erase(g, ids2erase):
    g.delete_edges(ids2erase)
