import igraph as ig
import networkx as nx

from edgeseraser import disparity


def test_nx_graph_filter():
    g = nx.erdos_renyi_graph(100, 0.1)
    disparity.filter_nx_graph(g)
    bb = nx.edge_betweenness_centrality(g, normalized=False)
    nx.set_edge_attributes(g, bb, "betweenness")
    disparity.filter_nx_graph(g, field="betweenness")


def test_ig_graph_filter():
    g = ig.Graph.Erdos_Renyi(100, 0.5)
    ne_old = g.ecount()
    disparity.filter_ig_graph(g, 0.1)
    assert ne_old > g.ecount()
    g.es["weight2"] = 1.0
    disparity.filter_ig_graph(g, 0.1, field="weight2")
    assert ne_old > g.ecount()
