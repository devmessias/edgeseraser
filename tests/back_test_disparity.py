import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt 


from edgeseraser import disparity



def test_nx_graph_filter():
    g = nx.erdos_renyi_graph(100, 0.1)
    ne_old = g.number_of_edges()
    disparity.filter_nx_graph(g, thresh=0.2)
    ne_new = g.number_of_edges()
    assert ne_old > ne_new
    g = nx.erdos_renyi_graph(100, 1/100, directed=True)
    g = g.subgraph(max(nx.weakly_connected_components(g), key=len))
    g = nx.Graph(g)
    nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())}, copy=False)
    bb = nx.edge_betweenness_centrality(g, normalized=True)
    nx.set_edge_attributes(g, bb, "betweenness")
    disparity.filter_nx_graph(g, field="betweenness", thresh=0.2)

    g = nx.erdos_renyi_graph(100, 0.1, directed=True)
    ne_old = g.number_of_edges()
    disparity.filter_nx_graph(g, thresh=0.2)
    ne_new = g.number_of_edges()
    assert ne_old > ne_new


def test_ig_graph_filter():
    g = ig.Graph.Erdos_Renyi(100, 0.5, directed=True)
    ne_old = g.ecount()
    disparity.filter_ig_graph(g, 0.1)
    assert ne_old > g.ecount()
    g = ig.Graph.Erdos_Renyi(100, 0.5)
    ne_old = g.ecount()
    g.es["weight2"] = 1.0
    disparity.filter_ig_graph(g, 0.1, field="weight2")
    assert ne_old > g.ecount()


# def test_sbm():
#     p_in = 0.5
#     p_out = 0.02
#     n = 100
#     sizes = [n, n]
#     p = [
#         [p_in, p_out],
#         [p_out, p_in],
#     ]
#     g = nx.stochastic_block_model(
#        sizes, p, seed=0, directed=False)
#     vc = len(g.nodes())

#     import numpy as np
#     for _ in range(300):
#         u = np.random.randint(0, 100)
#         v = np.random.randint(0, vc)
#         if u > 100:
#             continue
#         if u == v:
#             continue
#         if np.random.rand() < 0.5:
#             continue
#         if g.has_edge(u, v):
#             g.remove_edge(u, v)
#             continue
#         else:
#             g.add_edge(u, v)
    
#     g = g.to_directed()
#     data = {
#         (u, v): 1
#         for u, v in g.edges()
#     }
#     nx.set_edge_attributes(g, data, "weight")
#     #g = nx.erdos_renyi_graph(100, 10/100, directed=False)
#     g2 = g.copy()
#     eold = g.number_of_edges()
#     adj = nx.to_numpy_array(g)
#     disparity.filter_nx_graph(g, thresh=0.9,)
#     noise_score.filter_nx_graph(g2, 2.55)
#     enew = g.number_of_edges()
#     adj2 = nx.to_numpy_array(g)
#     adj3 = nx.to_numpy_array(g2)
#     fig, ax = plt.subplots(1, 3)
#     ax[0].set_title(f"edges {eold}")
#     ax[1].set_title(f"edges {enew}")
#     ax[0].imshow(adj)
#     ax[1].imshow(adj2)
#     ax[2].imshow(adj3)
#     plt.show()

# test_sbm()