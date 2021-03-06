from networkx.classes.graph import GraphFixType
from typing import Any
from typing import Generic
from typing import NewType
from typing import TypeVar


class DiGraph(GraphFixType):

    graph_attr_dict_factory: Any
    node_dict_factory: Any
    node_attr_dict_factory: Any
    adjlist_outer_dict_factory: Any
    adjlist_inner_dict_factory: Any
    edge_attr_dict_factory: Any
    graph: Any
    def __init__(self, incoming_graph_data: Any | None = ..., **attr) -> None: ...
    @property
    def adj(self): ...
    @property
    def succ(self): ...
    @property
    def pred(self): ...
    def add_node(self, node_for_adding, **attr) -> None: ...
    def add_nodes_from(self, nodes_for_adding, **attr) -> None: ...
    def remove_node(self, n) -> None: ...
    def remove_nodes_from(self, nodes) -> None: ...
    def add_edge(self, u_of_edge, v_of_edge, **attr) -> None: ...
    def add_edges_from(self, ebunch_to_add, **attr) -> None: ...
    def remove_edge(self, u, v) -> None: ...
    def remove_edges_from(self, ebunch) -> None: ...
    def has_successor(self, u, v): ...
    def has_predecessor(self, u, v): ...
    def successors(self, n): ...
    neighbors: Any
    def predecessors(self, n): ...
    @property
    def edges(self): ...
    out_edges: Any
    @property
    def in_edges(self): ...
    @property
    def degree(self): ...
    @property
    def in_degree(self): ...
    @property
    def out_degree(self): ...
    def clear(self) -> None: ...
    def clear_edges(self) -> None: ...
    def is_multigraph(self): ...
    def is_directed(self) -> bool: ...
    def to_undirected(self, reciprocal: bool = ..., as_view: bool = ...): ...
    def reverse(self, copy: bool = ...): ...
