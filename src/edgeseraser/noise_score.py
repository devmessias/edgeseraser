import warnings
from typing import Tuple, TypeVar

import networkx as nx  # type: ignore
import numpy as np
import scipy.sparse as sp  # type: ignore

warnings.simplefilter("ignore", FutureWarning)

NpOrFloat = TypeVar("NpOrFloat", np.ndarray, float)


def get_noise_score(
    st_u: NpOrFloat, st_v: NpOrFloat, vol: float, w: NpOrFloat
) -> Tuple[NpOrFloat, NpOrFloat]:
    """
    Get noise score for each edge

    Args:
        st_u: np.array
            strength for each vertex
        st_v: np.array
            strength for each vertex
        vol: float
            volume of the graph
        w: np.array
            edge weights

    Returns:
        (np.array, np.array):
        -   **scores_uv**
            Noise score for each edge.
        -   **std_uv**
            standard deviation of noise score for each edge

    """
    st_prod_uv = np.multiply(st_u, st_v)

    mean_prior_prob = st_prod_uv / (vol) ** 2.0
    kappa = vol / st_prod_uv
    score = (kappa * w - 1) / (kappa * w + 1)
    var_prior_prob = (
        (1 / vol**2.0)
        * (st_prod_uv * (vol - st_u) * (vol - st_v))
        / (vol**2 * (vol - 1))
    )

    alpha_prior = (mean_prior_prob**2 / var_prior_prob) * (
        1 - mean_prior_prob
    ) - mean_prior_prob

    beta_prior = (
        (mean_prior_prob / var_prior_prob) * (1 - mean_prior_prob**2.0)
        - 1
        + mean_prior_prob
    )

    alpha_post = alpha_prior + w
    beta_post = vol - w + beta_prior
    expected_puv = alpha_post / (alpha_post + beta_post)
    variance_uv = expected_puv * (1 - expected_puv) * vol
    d = 1 / (st_prod_uv) - vol * ((st_u + st_v) / st_prod_uv**2)
    variance_cuv = variance_uv * (2 * (kappa + w * d) / (kappa * w + 1) ** 2.0) ** 2.0
    sdev_cuv = variance_cuv**0.5

    return score, sdev_cuv


def filter_generic_graph(
    w_degree: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute noise corrected edge weights for a sparse graph.

    Args:
        w_degree: np.ndarray
            Weight degree of each vertex.
        edges: np.array
            Edges of the graph.
        weights: np.array
            Edge weights of the graph.

    Returns:
        (np.array, np.array):
        -   **scores_uv**
            Noise corrected edge weights.
        -   **std_uv**
            Standard deviation of noise corrected edge weights.

    """

    vol = w_degree.sum()

    st_u = w_degree[edges[:, 0]]
    st_v = w_degree[edges[:, 1]]
    scores = st_u * st_v
    ids_0 = np.argwhere(scores == 0)
    st_u[ids_0] = 1
    st_v[ids_0] = 1
    scores_uv, std_uv = get_noise_score(st_u, st_v, vol, weights)
    scores_uv[ids_0] = 0
    std_uv[ids_0] = 0.0
    return scores_uv, std_uv


def cond_noise_edges2erase(
    scores_uv: np.ndarray, std_uv: np.ndarray, thresh: float = 1.28
) -> np.ndarray:
    """Filter edges with high noise score.

    Args:
        scores_uv: np.array
            edge scores
        std_uv: np.array
            edge standard deviations
        thresh: float
            >Since this is roughly equivalent to a one-tailed test of
            statistical significance, common values of δ are 1.28, 1.64, and
            2.32, which approximate p-values of 0.1, 0.05, and 0.0

    Returns:
        np.array:
        indices of edges to be erased
    """
    ids2erase = np.argwhere(scores_uv <= thresh * std_uv).flatten()
    return ids2erase


def filter_nx_graph(g, thresh: float = 1.28, field: str = "weight") -> None:
    """Filter edge with high noise score from a networkx graph.

    Args:
        g: networkx.Graph
            Graph to be filtered.
        thresh: float
            >Since this is roughly equivalent to a one-tailed test of
            statistical significance, common values of δ are 1.28, 1.64, and
            2.32, which approximate p-values of 0.1, 0.05, and 0.0
        field: str
            Edge field to be used for filtering.

    Example:
        ```python
        import networkx as nx
        import edgeseraser as ee

        g = nx.erdos_renyi_graph(100, 0.1)
        ee.noise_score.filter_nx_graph(g, field=None)

        g # filtered graph
        ```

    """

    w_adj = nx.adjacency_matrix(g)
    w_degree = np.asarray(w_adj.sum(axis=1)).flatten().astype(np.float64)
    if field is None:
        edges = np.array([[u, v, 1.0] for u, v in g.edges()])
    else:
        edges = np.array([[u, v, d[field]] for u, v, d in g.edges(data=True)])
    weights = edges[:, 2].astype(np.float64)
    edges = edges[:, :2].astype(np.int64)

    scores_uv, std_uv = filter_generic_graph(w_degree, edges, weights)

    ids2erase = cond_noise_edges2erase(scores_uv, std_uv, thresh=thresh)
    g.remove_edges_from([(e[0], e[1]) for e in edges[ids2erase]])


def filter_ig_graph(g, thresh: float = 1.28, field: str = "weight") -> None:
    """Filter edge with high noise score from a igraph graph.

    Args:
        g: igraph.Graph
            Graph to be filtered.
        thresh: float
            >Since this is roughly equivalent to a one-tailed test of
            statistical significance, common values of δ are 1.28, 1.64, and
            2.32, which approximate p-values of 0.1, 0.05, and 0.0
        field: str
            Edge field to be used for filtering.

    Example:
        ```python
        import networkx as nx
        import edgeseraser as ee

        g = nx.erdos_renyi_graph(100, 0.1)
        ee.noise_score.filter_nx_graph(g, field=None)

        g # filtered graph
        ```

    """
    num_vertices = g.vcount()
    edges = np.array(g.get_edgelist())
    if field is None:
        weights = np.ones(edges.shape[0])
    else:
        weights = np.array(g.es[field]).astype(np.float64)

    w_adj = sp.csr_matrix((weights, edges.T), shape=(num_vertices, num_vertices))
    w_degree = np.asarray(w_adj.sum(axis=1)).flatten().astype(np.float64)

    scores_uv, std_uv = filter_generic_graph(w_degree, edges, weights)

    ids2erase = cond_noise_edges2erase(scores_uv, std_uv, thresh=thresh)
    g.delete_edges(ids2erase)
