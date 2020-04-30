import networkx as nx

class Genealogical(nx.DiGraph):

    @classmethod
    def from_digraph(cls, D):
        G = D.copy()
        G.generations = nx.dag_longest_path_length(G)
        return G

    
    @property
    def n_individuals(self):
        return len(self.nodes)

    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` to plot the genealogy"""
        pos = nx.drawing.nx_agraph.graphviz_layout(self, prog='dot')
        nx.draw(self, pos=pos, with_labels=labels, node_shape='s', ax=ax, font_color='white', font_size=8, **kwargs)
