import networkx as nx
from Genealogical import Genealogical
import numpy as np
import matplotlib.pyplot as plt



class Traversal(Genealogical):

    def __init__(self):
        super().__init__()

        
    def similarity(self):
        # A kinship-like distance function
        n = self.n_individuals        
        K = np.zeros((n,n), dtype=float)
        
        for i in range(n):
            K[i,i] = 0.5
            for j in range(i+1, n):
                if any(self.predecessors(j)):
                    p = next(self.predecessors(j))
                    K[i,j] = (K[i,p]/2)
                    K[j,i] = K[i,j]
        return K

    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        rev = self.reverse()
        pos = nx.drawing.nx_agraph.graphviz_layout(rev, prog='dot',
                                                   args='-Grankdir=BT')
        nx.draw(rev, pos=pos, with_labels=labels, node_shape='s', ax=ax, font_color='white', font_size=8, arrows=False, **kwargs)
