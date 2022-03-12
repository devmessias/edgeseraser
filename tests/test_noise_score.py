import networkx as nx
import numpy as np

from edgeseraser import noise_score


def test_noise_graph_filter():
    g = nx.erdos_renyi_graph(100, 0.1)
    noise_score.filter_nx_graph(g, field=None)
    bb = nx.edge_betweenness_centrality(g, normalized=False)
    nx.set_edge_attributes(g, bb, "betweenness")
    noise_score.filter_nx_graph(g, field="betweenness")


def test_get_noise_score():
    noise_score.get_noise_score(
        np.array([1, 2, 3]), np.array([1, 2, 3]), 10, np.array([1, 2, 3])
    )
