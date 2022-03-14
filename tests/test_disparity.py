import igraph as ig
import networkx as nx

from edgeseraser import disparity


def test_nx_graph_filter():
    g = nx.erdos_renyi_graph(100, 0.2)
    ne_old = g.number_of_edges()
    disparity.filter_nx_graph(g, thresh=0.4)
    ne_new = g.number_of_edges()
    assert ne_old > ne_new
    g = nx.erdos_renyi_graph(100, 1 / 100, directed=True)
    g = g.subgraph(max(nx.weakly_connected_components(g), key=len))
    g = nx.Graph(g)
    nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())}, copy=False)
    bb = nx.edge_betweenness_centrality(g, normalized=True)
    nx.set_edge_attributes(g, bb, "betweenness")
    disparity.filter_nx_graph(g, field="betweenness", thresh=0.2)

    g = nx.erdos_renyi_graph(100, 0.2, directed=True)
    ne_old = g.number_of_edges()
    disparity.filter_nx_graph(g, thresh=0.5)
    ne_new = g.number_of_edges()
    assert ne_old > ne_new


def test_ig_graph_filter():
    g = ig.Graph.Erdos_Renyi(100, 0.2, directed=True)
    g = g.components(mode="weak").giant()
    ne_old = g.ecount()
    disparity.filter_ig_graph(g, 0.9)
    assert ne_old > g.ecount()
    g = ig.Graph.Erdos_Renyi(100, 0.5)
    g = g.components(mode="weak").giant()
    ne_old = g.ecount()
    g.es["weight2"] = 1.0
    disparity.filter_ig_graph(g, 0.1, field="weight2")
