import igraph as ig
import networkx as nx
import numpy as np
import pytest
from edgeseraser import disparity, polya, polya_tools
from edgeseraser.misc import backend


def test_pdf():
    w = np.array([1, 2, 3, 4, 5])
    n = 40
    k = 10.0
    a = 0.0
    p = polya_tools.statistics.compute_polya_pdf_approx(w, n, k, a)
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


def test_igraph():
    g = ig.Graph.Erdos_Renyi(100, 0.6, directed=True)
    g = g.components(mode="weak").giant()
    polya.filter_ig_graph(g)


def test_integer_weights():
    eps = 10e-20
    g = nx.watts_strogatz_graph(100, 8, 0.2)
    g = g.subgraph(max(nx.connected_components(g), key=len))
    g = nx.Graph(g)
    nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())}, copy=False)
    bb = nx.edge_betweenness_centrality(g, normalized=False)
    bb = {(u, v): int(round(bb[(u, v)])) for u, v in g.edges()}
    nx.set_edge_attributes(g, bb, "betweenness")
    edges, weights, num_vertices, opts = backend.nx_extract(g, field="betweenness")
    assert np.mod(weights, 1).sum() < eps
    polya.scores_generic_graph(
        num_vertices, edges, weights, a=2, is_directed=opts["is_directed"]
    )


def large_graph_for_testing(n):
    g = ig.Graph.Erdos_Renyi(int(n), 3 / n, directed=False)
    g = g.components(mode="strong").giant()
    edges, _, num_vertices, opts = backend.ig_extract(g)
    return edges, num_vertices, opts


@pytest.mark.parametrize(
    "n, optimization",
    [
        (1 * 10e2, "lru-py-nb"),
        (1 * 10e2, "lru-nb"),
        (1 * 10e2, "lru-nbf"),
        (1 * 10e2, "lru-nb-szuszik"),
    ],
)
def test_lru_int_polya(n, optimization):
    wmax = 20
    ne = 10
    edges, num_vertices, opts = large_graph_for_testing(n)
    ne = len(edges)
    weights = np.random.randint(2, wmax, ne).astype(np.float64)
    assert np.mod(weights, 1).sum() == 0

    r_lru = polya.scores_generic_graph(
        **{
            "num_vertices": num_vertices,
            "edges": edges,
            "weights": weights,
            "is_directed": opts["is_directed"],
            "optimization": optimization,
        }
    )
    r = polya.scores_generic_graph(
        **{
            "num_vertices": num_vertices,
            "edges": edges,
            "weights": weights,
            "is_directed": opts["is_directed"],
            "optimization": "nb",
        }
    )
    r1 = r.copy()
    r1_lru = r_lru.copy()
    r1[r > 0] = 1
    r1_lru[r_lru > 0] = 1
    assert np.allclose(r1, r1_lru)
    assert np.allclose(r, r_lru)


@pytest.mark.parametrize(
    "n, wmax, optimization",
    [
        (1 * 10e3, 6, "lru-py-nb"),
        (1 * 10e3, 6, "lru-nb"),
        (1 * 10e3, 6, "lru-nbf"),
        (1 * 10e3, 6, "lru-nb-szuszik"),
        (1 * 10e3, 6, "nb"),
        (1 * 10e3, 30, "lru-py-nb"),
        (1 * 10e3, 30, "lru-nb"),
        (1 * 10e3, 30, "lru-nbf"),
        (1 * 10e3, 30, "lru-nb-szuszik"),
        (1 * 10e3, 30, "nb"),
    ],
)
@pytest.mark.benchmark(group="Polya-Urn: Int weights")
def test_polya_int_weights_perf(benchmark, n, wmax, optimization):
    edges, num_vertices, opts = large_graph_for_testing(n)
    weights = np.random.randint(1, wmax, len(edges)).astype(np.float64)
    benchmark.extra_info["Num Edges"] = len(edges)
    benchmark.extra_info["Num_vertices"] = num_vertices
    assert np.mod(weights, 1).sum() == 0
    result = benchmark.pedantic(
        polya.scores_generic_graph,
        kwargs={
            "num_vertices": num_vertices,
            "edges": edges,
            "weights": weights,
            "is_directed": opts["is_directed"],
            "optimization": optimization,
        },
    )
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.parametrize("n", [(10e4), (5 * 10e4)])
@pytest.mark.benchmark(group="Polya-Urn: Float Weights")
def test_polya_float_weights_perf(benchmark, n):
    edges, num_vertices, opts = large_graph_for_testing(n)
    benchmark.extra_info["Num Edges"] = len(edges)
    benchmark.extra_info["Num_vertices"] = num_vertices
    weights = np.random.uniform(0, 1, len(edges))
    result = benchmark.pedantic(
        polya.scores_generic_graph,
        kwargs={
            "num_vertices": num_vertices,
            "edges": edges,
            "weights": weights,
            "is_directed": opts["is_directed"],
        },
    )
    assert np.all((result >= 0) & (result <= 1))
