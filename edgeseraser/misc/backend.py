from typing import Optional, Tuple, Union

import igraph as ig
import networkx as nx
import numpy as np
from edgeseraser.misc.typing import (
    IgOpts,
    NpArrayEdges,
    NpArrayEdgesFloat,
    NpArrayEdgesIds,
    NxOpts,
)


def nx_extract(
    g: Union[nx.Graph, nx.DiGraph],
    remap_labels: bool = False,
    field: Optional[str] = None,
) -> Tuple[NpArrayEdges, NpArrayEdgesFloat, int, NxOpts]:
    is_directed = g.is_directed()
    if is_directed:
        assert nx.is_weakly_connected(g), "Graph is not connected"
    else:
        assert nx.is_connected(g), "Graph is not connected"

    nodes = g.nodes()
    num_vertices = len(nodes)
    opts: NxOpts = {"is_directed": is_directed, "nodelabel2index": {}}
    try:
        nodes[0]
    except KeyError:
        remap_labels = True
    if remap_labels:
        nodelabel2index = {node: i for i, node in enumerate(nodes)}
        nx.relabel_nodes(g, nodelabel2index, copy=False)
        opts["nodelabel2index"] = nodelabel2index
    edges: NpArrayEdges = np.array([[u, v] for u, v in g.edges()])
    if field is None:
        weights = np.ones(edges.shape[0])
    else:
        weights = np.array([d[field] for u, v, d in g.edges(data=True)]).astype(
            "float64"
        )
    assert edges.shape[0] > 0, "Graph is empty"
    return edges, weights, num_vertices, opts


def nx_erase(
    g: Union[nx.Graph, nx.DiGraph], edges2erase: NpArrayEdgesIds, opts: NxOpts
) -> None:
    g.remove_edges_from([(e[0], e[1]) for e in edges2erase])
    nodelabel2index = opts.get("nodelabel2index")
    if nodelabel2index:
        nx.relabel_nodes(
            g, {i: node for node, i in nodelabel2index.items()}, copy=False
        )


def ig_extract(
    g: ig.Graph,
    field: Optional[str] = None,
) -> Tuple[NpArrayEdges, NpArrayEdgesFloat, int, IgOpts]:
    is_directed = g.is_directed()
    if is_directed:
        components = g.components(mode="weak")
        assert len(components) == 1, "Graph is not connected"
    else:
        assert g.is_connected(), "Graph is not connected"

    num_vertices = g.vcount()
    opts: IgOpts = {"is_directed": is_directed}
    edges: NpArrayEdges = np.array(g.get_edgelist())
    assert edges.shape[0] > 0, "Graph is empty"
    if field is None:
        weights = np.ones(edges.shape[0])
    else:
        weights = np.array(g.es[field]).astype(np.float64)
    return edges, weights, num_vertices, opts


def ig_erase(g: ig.Graph, ids2erase: NpArrayEdgesIds) -> None:
    g.delete_edges(ids2erase)
