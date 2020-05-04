import networkx as nx
from collections.abc import Iterable


class Genealogical(nx.DiGraph):

    # @classmethod
    # def from_digraph(cls, D):
    #     G = D.copy()
    #     return G

    @property
    def generations(self):
        return nx.dag_longest_path_length(self)
    
    @property
    def n_individuals(self):
        return len(self.nodes)

    def filter_nodes(self, predicate):
        node_list = []
        for node, data in self.nodes(data=True):
            if predicate(node, data):
                node_list.append(node)
        return node_list

    def node_attribute(self, attribute):
        return (node[1][attribute] for node in self.nodes(data=True))

    def founders(self, use_time=True):
        """Get a list of individuals at time 0"""
        if use_time:
            return self.filter_nodes(
                lambda node, data: data['time'] == 0)
        else:
            return self.filter_nodes(
                lambda node, data: len(list(self.predecessors(node))) == 0
            )

    def probands(self, use_time=True):
        """Get a list of individuals at present day"""
        if use_time:
            return self.filter_nodes(
                lambda node, data: data['time'] == self.generations)
        else:
            return self.filter_nodes(
                lambda node, data: len(list(self.successors(node))) == 0
            )

    def predecessors_at_k(self, node, k=1):
        """
        Get a list of all predecessors that are up to `k` steps away from node `node`.
        If we hit a founder before `k`, include that founder in the list of predecessors.
        """

        nodes = [node]
        while k > 0:
            new_nodes = []

            for n in nodes:
                pred_list = list(self.predecessors(n))
                if len(pred_list) == 0:
                    new_nodes.append(n)
                else:
                    new_nodes += pred_list

            nodes = new_nodes
            k -= 1

        return nodes

    def get_probands_under(self, nodes=None, climb_up_step=0):

        if nodes is None:
            nodes = list(self.nodes())
        elif not isinstance(nodes, Iterable) or type(nodes) == str:
            nodes = [nodes]

        ntp = {}  # Nodes to probands

        for n in nodes:

            ntp[n] = set()

            base_set = self.predecessors_at_k(n, climb_up_step)
            n_set = []
            for ns in base_set:
                n_set += list(self.successors(ns))

            if len(n_set) == 0:
                ntp[n].add(n)
            else:
                while len(n_set) > 0:
                    nn, nn_children = n_set[0], list(self.successors(n_set[0]))

                    if len(nn_children) > 0:
                        n_set.extend(nn_children)
                    else:
                        ntp[n].add(nn)

                    del n_set[0]

        return ntp

    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        pos = nx.drawing.nx_agraph.graphviz_layout(self, prog='dot')
        nx.draw(self, pos=pos, with_labels=labels, node_shape='s', ax=ax, font_color='white', font_size=8, **kwargs)
