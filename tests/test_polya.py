import networkx as nx
import numpy as np

from edgeseraser import disparity, polya
from edgeseraser.misc import backend


def test_pdf():
    w = np.array([1, 2, 3, 4, 5])
    n = 40
    k = 10.0
    a = 0.0
    p = polya.compute_polya_pdf(w, n, k, a)
    assert np.allclose(np.argsort(p), np.array([0, 1, 2, 3, 4]))


def test_polya_scores_disp_limit():
    g = nx.erdos_renyi_graph(100, 6 / 100, directed=True)
    g = g.subgraph(max(nx.weakly_connected_components(g), key=len))
    g = nx.Graph(g)
    nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())}, copy=False)
    data = {(u, v): np.random.uniform(0.1, 0.9) for u, v in g.edges()}
    nx.set_edge_attributes(g, data, "weight")

    edges, weights, num_vertices, opts = backend.nx_extract(g, field="weight")
    p1 = polya.scores_generic_graph(num_vertices, edges, weights, a=1)
    p_disparity = disparity.scores_generic_graph(num_vertices, edges, weights)
    np.testing.assert_almost_equal(p1, p_disparity)


def test_polya_scores_int_weight():
    g = nx.circulant_graph(10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    data = {(u, v): np.random.uniform(1, 10) for u, v in g.edges()}
    nx.set_edge_attributes(g, data, "weight")
    polya.filter_nx_graph(g, field="weight")
