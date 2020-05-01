import networkx as nx

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
        return (node[1]['time'] for node in self.nodes(data=True))

    
    def founders(self):
        """Get a list of individuals at time 0"""
        return self.filter_nodes(
            lambda node, data: data['time'] == 0)
    

    def probands(self):
        """Get a list of individuals at present day"""
        return self.filter_nodes(
            lambda node, data: data['time'] == self.generations)


    def draw(self, labels=True, ax=None, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        pos = nx.drawing.nx_agraph.graphviz_layout(self, prog='dot')
        nx.draw(self, pos=pos, with_labels=labels, node_shape='s', ax=ax, font_color='white', font_size=8, **kwargs)
