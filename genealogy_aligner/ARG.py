from .Genealogical import Genealogical
import networkx as nx


class ARG(Genealogical):
    """
    The ARG class is a sequence of Traversal Objects.
    It provides functionalities for fusing and manipulating
    multiple Traversal objects.
    """

    def __init__(self, traversals=None, graph_type='union'):

        super().__init__()
        self.graph_type = graph_type
        self.traversals = []

        if traversals is not None:
            for tr in traversals:
                self.add_traversal(tr)

    def add_traversal(self, tr):
        """
        TODO: Need to handle negative nodes for union graphs
        """

        tr_id = len(self.traversals)

        if self.graph_type == 'union':
            self.graph.add_edges_from(tr.edges)

        elif self.graph_type == 'fused':
            self.graph = nx.union(self.graph, tr.graph, rename=(None, f'TR{tr_id}-'))

            if len(self.traversals) > 0 and tr_id > 0:
                self.graph.add_edges_from([(f'TR{tr_id - 1}-{proband}', f'TR{tr_id}-{proband}')
                                           for proband in tr.probands()], label='fuse')
        else:
            raise KeyError(f"Graph type {self.graph_type} is not supported.")

        self.traversals.append(tr)

    def set_graph_type(self, graph_type):
        self.graph_type = graph_type
        self.graph = nx.DiGraph()

        current_traversals = self.traversals.copy()
        self.traversals = []

        for tr in current_traversals:
            self.add_traversal(tr)

    def __len__(self):
        return len(self.traversals)

    def __iter__(self):

        for tr in self.traversals:
            yield tr
