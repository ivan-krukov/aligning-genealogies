import networkx as nx
import numpy as np
from collections.abc import Iterable


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
            return node_attr[node]


    def attribute_array(self, nodes, attr):
        """Return a sorted array of attributes for the given nodes"""
        attr_dict = nx.get_node_attributes(self.graph, attr)
        return np.array([attr_dict[n] for n in nodes])


    def get_individuals_at_generation(self, k):
        return self.filter_nodes(lambda node, data: data['time'] == k)

    
    def founders(self, use_time=False):
        """
        Get a list of nodes that don't have predecessors
        :return:
        """
        if use_time:
            return self.filter_nodes(
                lambda node, data: data['time'] == self.generations
            )
        else:
            return self.filter_nodes(
                lambda node, data: len(list(self.graph.predecessors(node))) == 0
            )

    def probands(self, use_time=True):
        """Get a list of individuals at present day"""
        if use_time:
            return self.filter_nodes(
                lambda node, data: data['time'] == 0
            )
        else:
            return self.filter_nodes(
                lambda node, data: len(list(self.graph.successors(node))) == 0
            )

    def get_probands_under(self, nodes=None, climb_up_step=0):

        if nodes is None:
            nodes = self.nodes
        elif not isinstance(nodes, Iterable) or type(nodes) == str:
            nodes = [nodes]

        ntp = {}  # Nodes to probands

        for n in nodes:

            ntp[n] = set()

            base_set = self.predecessors(n, climb_up_step, include_self=True)
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

    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos=pos, with_labels=labels,
                node_shape='s', ax=ax, font_color='white', font_size=8, **kwargs)
