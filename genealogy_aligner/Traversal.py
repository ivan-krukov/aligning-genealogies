import networkx as nx
from Genealogical import Genealogical
import numpy as np
import copy


class Traversal(Genealogical):

    def __init__(self):
        super().__init__()
        
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

        if inplace:
            t_obj = self
        else:
            t_obj = copy.deepcopy(self)

        non_coalesc_nodes = t_obj.filter_nodes(
                lambda node, data: len(t_obj.successors(node)) == 1
        )

        edges_to_add = []

        for n in non_coalesc_nodes:

            pred_n = list(t_obj.predecessors(n))

            if len(pred_n) > 0:
                pred_n = pred_n[0]
                if pred_n not in non_coalesc_nodes:
                    k = n
                    edge_weight = 1
                    while k in non_coalesc_nodes:
                        k = t_obj.successors(k)[0]
                        edge_weight += 1
                    edges_to_add.append((pred_n, k, dict(dist=edge_weight)))

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
                    tree_founders = [f for f in tree_founders if f != n]
                tree_founders.append(ca_counter)

        t_obj.graph.remove_nodes_from(list(nx.isolates(t_obj.graph)))

        if not inplace:
            return t_obj

    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        rev = self.graph.reverse()
        pos = nx.drawing.nx_agraph.graphviz_layout(rev, prog='dot',
                                                   args='-Grankdir=BT')
        nx.draw(rev, pos=pos, with_labels=labels, node_shape='s', ax=ax,
                font_color='white', font_size=8, arrows=False, **kwargs)
