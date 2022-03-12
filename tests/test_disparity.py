import networkx as nx

from edgeseraser import disparity


def test_noise_graph_filter():
    g = nx.erdos_renyi_graph(100, 0.1)
    disparity.filter_nx_graph(g, field=None)
    bb = nx.edge_betweenness_centrality(g, normalized=False)
    nx.set_edge_attributes(g, bb, "betweenness")
    disparity.filter_nx_graph(g, field="betweenness")
