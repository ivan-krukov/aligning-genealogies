import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .Drawing import get_graphviz_layout
from .Genealogical import Genealogical


class DiploidGraph(Genealogical):

    def __init__(self, P):

        D = nx.DiGraph()
        super().__init__(D)
        
        sex = P.get_node_attributes('sex')
        
        for trio in P.iter_trios():
            father, mother, child = trio.values()

            f_pat, f_mat = 2 * father - 1, 2 * father  # paternal ploids
            m_pat, m_mat = 2 * mother - 1, 2 * mother  # maternal ploids
            c_pat, c_mat = 2 * child  - 1, 2 * child   # offspring ploids

            self.graph.add_nodes_from([(f_pat, {'individual': father, 'sex': 1}),
                              (f_mat, {'individual': father, 'sex': 1}),
                              (c_pat, {'individual': child,  'sex': sex[child]})])
            self.graph.add_edges_from([(f_pat, c_pat), (f_mat, c_pat)])

            self.graph.add_nodes_from([(m_pat, {'individual': mother, 'sex': 2}),
                              (m_mat, {'individual': mother, 'sex': 2}),
                              (c_mat, {'individual': child,  'sex': sex[child]})])
            self.graph.add_edges_from([(m_pat, c_mat), (m_mat, c_mat)])

            

    def draw(self, ax=None, nudge=30, figsize=(8,6), node_size=800):

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        sex = nx.get_node_attributes(self.graph, 'sex')
        
        pos = get_graphviz_layout(self.graph)

        pos_left_nudge = {node: (x-nudge, y) for node, (x,y) in pos.items()}

        individuals = nx.get_node_attributes(self.graph, 'individual')

        males = [node for node, ind in individuals.items() if sex[node] == 1]
        nx.draw_networkx_nodes(self.graph, pos=pos, ax=ax, node_shape='s',
                               node_size=node_size, nodelist=males)
        nx.draw_networkx_labels(self.graph, pos=pos, ax=ax, nodelist=males,
                                font_size=12, font_color='white')

        females = [node for node, ind in individuals.items() if sex[node] == 2]
        nx.draw_networkx_nodes(self.graph, pos=pos, ax=ax, node_shape='o',
                               node_size=node_size, nodelist=females)
        nx.draw_networkx_labels(self.graph, pos=pos, ax=ax, nodelist=females,
                                font_size=12, font_color='white')

        
        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax, node_size=node_size)

        nx.draw_networkx_labels(self.graph, pos_left_nudge, ax=ax,
                                labels = individuals, font_color='firebrick', font_size=12)


        subscript_patch = mpatches.Patch(color='firebrick', label='Individual ID')

        ax.legend(handles=[subscript_patch], loc='lower right')

        return ax


        
