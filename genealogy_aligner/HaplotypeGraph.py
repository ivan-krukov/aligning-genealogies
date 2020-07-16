import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import count

from numpy import random as rnd

from .Drawing import get_graphviz_layout
from .Genealogical import Genealogical
from .Traversal import Traversal


class HaplotypeGraph(Genealogical):

    def __init__(self, ped, graph_type="autosomal"):

        super().__init__()

        self.pedigree = ped
        self.generations = ped.generations
        self.graph_type = graph_type

        self.ind_to_hap = None

        if graph_type == "autosomal":
            self.init_autosomal_graph()
        elif graph_type == "mitochondrial":
            self.init_mitochondrial_graph()
        elif graph_type == "X":
            self.init_x_graph()
        elif graph_type == "Y":
            self.init_y_graph()
        else:
            raise NotImplementedError(f"Haplotype graph type {graph_type} is not implemented")

    def pairs(self, node):
        """
        For a given haplotype, find its pair(s).
        """

        pairs = list(set([
            parent for child in self.successors(node)
            for parent in self.predecessors(child) if parent != node
        ]))

        if len(pairs) == 0:
            return [0]
        else:
            return pairs

    def init_autosomal_graph(self):
        """
        Initialize the haplotype graph assuming autosomal inheritance pattern.
        """

        ind_time = self.pedigree.get_node_attributes('time')

        hap_dict = {}

        for i, n in enumerate(self.pedigree.nodes, 1):

            hap_dict[n] = [2 * i - 1, 2*i]

            for h_idx in hap_dict[n]:
                self.graph.add_node(h_idx, individual=n, time=ind_time[n])

        for n in hap_dict:

            for h_idx, parent in zip(hap_dict[n], self.pedigree.get_parents(n)):

                # If the parent is a node in the pedigree:
                if parent in hap_dict:
                    self.graph.add_edges_from([(p_h_idx, h_idx)
                                               for p_h_idx in hap_dict[parent]])

        self.ind_to_hap = hap_dict

    def init_y_graph(self):
        """
        Initialize the haplotype graph assuming Chr Y inheritance pattern
        """

        assert "sex" in self.pedigree.attributes

        ind_time = self.pedigree.get_node_attributes('time')
        ind_sex = self.pedigree.get_node_attributes('sex')

        hap_dict = {}
        hap_counter = count(1)

        for n in self.pedigree.nodes:

            if ind_sex[n] == 1:
                hap_dict[n] = next(hap_counter)
                self.graph.add_node(hap_dict[n], individual=n, time=ind_time[n])

        for n in hap_dict:

            father, mother = self.pedigree.get_parents(n)

            if father in hap_dict:
                self.graph.add_edge(hap_dict[father], hap_dict[n])

        self.ind_to_hap = hap_dict

    def init_mitochondrial_graph(self):
        """
        Initialize the haplotype graph assuming mitochondrial inheritance pattern
        """

        assert "sex" in self.pedigree.attributes

        ind_time = self.pedigree.get_node_attributes('time')

        hap_dict = {}

        for i, n in enumerate(self.pedigree.nodes, 1):
            hap_dict[n] = i
            self.graph.add_node(i, individual=n, time=ind_time[n])

        for n in hap_dict:

            father, mother = self.pedigree.get_parents(n)

            if mother in hap_dict:
                self.graph.add_edge(hap_dict[mother], hap_dict[n])

        self.ind_to_hap = hap_dict

    def init_x_graph(self):

        assert "sex" in self.pedigree.attributes

        ind_time = self.pedigree.get_node_attributes('time')
        ind_sex = self.pedigree.get_node_attributes('sex')

        hap_dict = {}
        hap_counter = count(1)

        for n in self.pedigree.nodes:

            if ind_sex[n] == 1:
                hap_dict[n] = [next(hap_counter)]
            elif ind_sex[n] == 2:
                hap_dict[n] = [next(hap_counter), next(hap_counter)]

            for h_idx in hap_dict[n]:
                self.graph.add_node(h_idx, individual=n, time=ind_time[n])

        for n in hap_dict:

            for h_idx, parent in zip(hap_dict[n], self.pedigree.get_parents(n)[::-1]):
                if parent in hap_dict:
                    self.graph.add_edges_from([(p_h_idx, h_idx)
                                               for p_h_idx in hap_dict[parent]])

        self.ind_to_hap = hap_dict

    def sample_path(self, haploid_probands=False):

        assert self.graph_type in ('autosomal', 'X')

        hap_time = self.get_node_attributes('time')

        tr = Traversal()
        tr.haploid_probands = haploid_probands

        nodes_to_climb = []

        for proband in self.pedigree.probands():
            proband_haps = self.ind_to_hap[proband]
            if haploid_probands:
                proband_haps = rnd.choice(proband_haps, 1)

            for h_idx in proband_haps:
                tr.graph.add_node(h_idx, time=hap_time[h_idx])
                nodes_to_climb.append(h_idx)

        while len(nodes_to_climb) > 0:

            curr_node, nodes_to_climb = nodes_to_climb[0], nodes_to_climb[1:]

            if tr.predecessors(curr_node):
                continue

            try:
                parent = rnd.choice(self.predecessors(curr_node))
            except ValueError:
                continue

            if parent not in tr.nodes:
                tr.graph.add_node(parent, time=hap_time[parent])

            tr.graph.add_edge(parent, curr_node)

            if len(self.predecessors(parent)) > 0 and len(tr.predecessors(parent)) == 0:
                nodes_to_climb.append(parent)

        tr.ts_node_to_ped_node = dict(zip(tr.nodes, tr.nodes))

        return tr

    """
    def draw(self, ax=None, nudge=30, figsize=(8, 6), node_size=800):

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        sex = self.pedigree.get_node_attributes('sex')
        pos = get_graphviz_layout(self.graph)
        pos_left_nudge = {node: (x - nudge, y) for node, (x, y) in pos.items()}
        pos_right_nudge = {node: (x + nudge, y) for node, (x, y) in pos.items()}
        individuals = self.get_node_attributes("individual")
        times = self.get_node_attributes("time")

        print(sex)

        males = [node for node, ind in individuals.items() if sex[ind] == 1]
        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=ax,
            node_shape="s",
            node_size=node_size,
            nodelist=males,
        )
        nx.draw_networkx_labels(
            self.graph,
            pos=pos,
            ax=ax,
            nodelist=males,
            font_size=12,
            font_color="white",
            labels=individuals,
        )

        females = [node for node, ind in individuals.items() if sex[ind] == 2]
        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            ax=ax,
            node_shape="o",
            node_size=node_size,
            nodelist=females,
        )
        nx.draw_networkx_labels(
            self.graph,
            pos=pos,
            ax=ax,
            nodelist=females,
            font_size=12,
            font_color="white",
            labels=individuals,
        )

        nx.draw_networkx_edges(self.graph, pos=pos, ax=ax, node_size=node_size)
        nx.draw_networkx_labels(
            self.graph, pos_left_nudge, ax=ax, font_color="firebrick", font_size=12
        )
        nx.draw_networkx_labels(
            self.graph,
            pos_right_nudge,
            ax=ax,
            font_color="forestgreen",
            font_size=12,
            labels=times,
        )

        left_patch = mpatches.Patch(color="firebrick", label="Ploid ID")
        right_patch = mpatches.Patch(color="forestgreen", label="Generation")
        ax.legend(handles=[left_patch, right_patch], loc="lower right")

        return ax
    """
