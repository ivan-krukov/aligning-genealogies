import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt

from .Genealogical import Genealogical


class Traversal(Genealogical):

    def __init__(self):
        super().__init__()
        self.ts_edges_to_ped_nodes = {}
        
    def similarity(self, G):
        # A kinship-like distance function
        n = G.n_individuals
        K = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            if i in self:
                K[i, i] = 0.5
                for j in range(i+1, n):
                    if j in self:
                        if any(self.predecessors(j)):
                            p = next(self.predecessors(j))
                            K[i, j] = (K[i, p]/2)
                            K[j, i] = K[i, j]
        return K

    def to_coalescent_tree(self, add_common_ancestors=True, inplace=False):

        t_obj = copy.deepcopy(self)

        non_coalesc_nodes = t_obj.filter_nodes(
                lambda node, data: len(t_obj.successors(node)) == 1
        )

        edges_to_add = []

        for n in non_coalesc_nodes:

            pred_n = t_obj.predecessors(n)

            if len(pred_n) > 0:
                pred_n = pred_n[0]
                if pred_n not in non_coalesc_nodes:
                    k = n
                    edge_weight = 1
                    ped_nodes = []
                    while k in non_coalesc_nodes:
                        ped_nodes.append(k)
                        k = t_obj.successors(k)[0]
                        edge_weight += 1

                    edges_to_add.append((pred_n, k, dict(dist=edge_weight)))
                    t_obj.ts_edges_to_ped_nodes[(pred_n, k)] = ped_nodes

        t_obj.graph.add_edges_from(edges_to_add)
        t_obj.graph.remove_nodes_from(non_coalesc_nodes)

        tree_founders = t_obj.founders()

        if add_common_ancestors:
            ca_counter = 0
            while len(tree_founders) > 1:
                ca_counter -= 1
                nodes_to_merge = np.random.choice(tree_founders, size=2, replace=False)
                for n in nodes_to_merge:

                    t_obj.graph.add_edge(ca_counter, n)

                    if n >= 0 and n not in self.founders():

                        k = self.predecessors(n)[0]
                        ped_nodes = [k]

                        while k not in self.founders():
                            k = self.predecessors(k)[0]
                            ped_nodes.append(k)

                        t_obj.ts_edges_to_ped_nodes[(ca_counter, n)] = ped_nodes

                    tree_founders = [f for f in tree_founders if f != n]
                tree_founders.append(ca_counter)

        t_obj.graph.remove_nodes_from(list(nx.isolates(t_obj.graph)))

        # Set the time attribute for out-of-pedigree nodes to inf for now:
        nx.set_node_attributes(t_obj.graph,
                               {ind: np.inf for ind in t_obj.nodes if int(ind) < 0},
                               'time')

        if inplace:
            self = t_obj
        else:
            return t_obj

    def get_graphviz_layout(self):
        return nx.drawing.nx_agraph.graphviz_layout(self.graph.reverse(),
                                                    prog='dot',
                                                    args='-Grankdir=BT')

    def draw(self, ax=None, figsize=(8, 6),
             node_color=None, labels=True, node_shape='s',
             default_color='#2b8cbe', **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if node_color is None:
            node_col = [default_color]*self.n_individuals
        else:
            node_col = []
            for n in self.nodes:
                try:
                    node_col.append(node_color[n])
                except KeyError:
                    node_col.append(default_color)

        nx.draw(self.graph.reverse(),
                pos=self.get_graphviz_layout(),
                with_labels=labels, node_shape=node_shape,
                node_color=node_col, ax=ax, font_color='white', font_size=8,
                arrows=False, **kwargs)
