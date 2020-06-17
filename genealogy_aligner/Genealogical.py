import networkx as nx
from collections.abc import Iterable

from .Drawing import draw
import numpy as np


class Genealogical(object):

    def __init__(self, graph=None):
        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph

    @property
    def n_individuals(self):
        return len(self.nodes)

    @property
    def edges(self):
        return list(self.graph.edges())

    @property
    def nodes(self):
        return list(self.graph.nodes())

    @property
    def attributes(self):
        return list(list(self.graph.nodes(data=True))[0][1].keys())

    def get_edge_attributes(self, attr):
        return nx.get_edge_attributes(self.graph, attr)

    def predecessors(self, node, k=1, include_founders=False):

        nodes = [node]
        while k > 0:
            new_nodes = []

            for n in nodes:
                pred_list = list(self.graph.predecessors(n))
                if len(pred_list) == 0 and include_founders:
                    new_nodes.append(n)
                else:
                    new_nodes += pred_list

            nodes = new_nodes
            k -= 1

        return nodes

    def successors(self, node, k=1, include_leaves=False):

        nodes = [node]
        while k > 0:
            new_nodes = []

            for n in nodes:
                succ_list = list(self.graph.successors(n))
                if len(succ_list) == 0 and include_leaves:
                    new_nodes.append(n)
                else:
                    new_nodes += succ_list

            nodes = new_nodes
            k -= 1

        return nodes

    def filter_nodes(self, predicate):
        node_list = []
        for node, data in self.graph.nodes(data=True):
            if predicate(node, data):
                node_list.append(node)
        return node_list

    def get_node_attributes(self, attr, node=None):

        node_attr = nx.get_node_attributes(self.graph, attr)

        if node is None:
            return node_attr
        else:
            try:
                return node_attr[node]
            except KeyError:
                return {}

    def nodes_at_generation_view(self, k):
        time = self.get_node_attributes('time')
        G = self.graph
        return nx.subgraph_view(G, lambda n: time[n] == k)

    def nodes_at_generation(self, k):
        return list(self.nodes_at_generation_view(k).nodes)        

    def founders_view(self):
        G = self.graph
        return nx.subgraph_view(G, lambda n: not any(G.predecessors(n)))

    def founders(self):
        """
        Get a list of nodes that don't have predecessors
        """
        return list(self.founders_view().nodes)

    def probands_view(self):
        G = self.graph
        return nx.subgraph_view(G, lambda n: not any(G.successors(n)))
        
    def probands(self):
        """Get a list of individuals with no children"""
        return list(self.probands_view().nodes)

    def trace_edges(self, forward=True, source=None):
        """Trace edges in a breadth-first-search
        Yields a pair of `(node, neighbor)`
        Note that the same edge can appear in the tracing more than once
        For `forward=True` iteration, start at founders (nodes with no predecessors), and yield `(node, child)` pairs
        For `forward=False` iteration, start at probands (nodes with no successors), and yeild `(node, parent)` pairs.
        Optional `source` argument can be used to specify the starting nodes
        
        Parameters
        ----------
        forward: boolean
            Direction of iteration:
            - `Forward=True` - from parents to children, yielding `(node, child)` pairs
            - `Forward=False` - from children to parents, , yielding `(node, parent)` pairs
        source: [int]
            Iterable of node IDs to initialize the iteration. By default, direction-specific source nodes are chosen
        
        Yields
        -------
        pair of `(node, neighbor)` node IDs, for each edge
        """
        if (source is None) and forward:
            source = self.founders_view().nodes
        elif (source is None) and (not forward):
            source = self.probands_view().nodes

        neighbors = self.graph.successors if forward else self.graph.predecessors
        
        curr_gen = set(source)
        next_gen = set()

        while curr_gen:
            for node in curr_gen:
                for neighbor in neighbors(node):
                    yield node, neighbor
                    next_gen.add(neighbor)
            curr_gen = next_gen
            next_gen = set()

    def get_probands_under(self, nodes=None, climb_up_step=0):

        if nodes is None:
            nodes = self.nodes
        elif not isinstance(nodes, Iterable) or type(nodes) == str:
            nodes = [nodes]

        ntp = {}  # Nodes to probands

        for n in nodes:

            ntp[n] = set()

            base_set = self.predecessors(n, climb_up_step, include_founders=True)
            n_set = []
            for ns in base_set:
                n_set += self.successors(ns)

            if len(n_set) == 0:
                ntp[n].add(n)
            else:
                while len(n_set) > 0:
                    nn, nn_children = n_set[0], self.successors(n_set[0])

                    if len(nn_children) > 0:
                        n_set.extend(nn_children)
                    else:
                        ntp[n].add(nn)

                    del n_set[0]

        return ntp

    def draw(self, **kwargs):
        return draw(self.graph, **kwargs)

    def similarity(self):
        # A kinship-like distance function
        n = self.n_individuals        
        K = np.zeros((n,n), dtype=float)

        for i in range(n):
            K[i,i] = 0.5
            for j in range(i+1, n):
                # this should not be necessary
                if any(self.graph.predecessors(j)):
                    p = next(self.graph.predecessors(j))
                    K[i,j] = (K[i,p]/2)
                    K[j,i] = K[i,j]
        return K
