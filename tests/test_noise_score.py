import igraph as ig
import networkx as nx
import numpy as np

from edgeseraser import noise_score


def test_nx_filter():
    g = nx.circulant_graph(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    noise_score.filter_nx_graph(g)
    bb = nx.edge_betweenness_centrality(g, normalized=False)
    nx.set_edge_attributes(g, bb, "betweenness")
    noise_score.filter_nx_graph(g, field="betweenness")
    g = nx.Graph()
    g.add_nodes_from([chr(i) for i in range(100)])
    g.add_edges_from([(chr(i), chr(i + 1)) for i in range(99)])
    noise_score.filter_nx_graph(g)


def test_get_noise_score():
    noise_score.noisy_scores(
        np.array([1, 2, 3]), np.array([1, 2, 3]), 10, np.array([1, 2, 3])
    )


def test_ig_graph_filter():
    g = ig.Graph.Erdos_Renyi(100, 1, directed=False)
    cl = g.clusters()
    g = cl.giant()
    ne_old = g.ecount()
    g2 = g.copy()
    noise_score.filter_ig_graph(g, 0.1)
    assert ne_old > g.ecount()
    g2.es["weight2"] = 1.0
    noise_score.filter_ig_graph(g2, 0.1, field="weight2")
    assert ne_old > g2.ecount()
    g = ig.Graph()
    for i in range(100):
        g.add_vertex(name=chr(i))
    for i in range(99):
        g.add_edge(chr(i), chr(i + 1))
    noise_score.filter_ig_graph(g, 0.1)
