from itertools import count, product

import pandas as pd
import numpy as np
from numpy import random as rnd
import datetime
from scipy import sparse

from .Genealogical import Genealogical
from .Traversal import Traversal
from .utils import dict_to_csr, invert_dictionary


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

    def get_individual_attributes(self, attr, node=None):
        """
        Get the attributes of the individual assigned to each haplotype.
        """

        ind_attr = self.pedigree.get_node_attributes(attr)

        if node is None:
            ind_dict = self.get_node_attributes('individual')
            return {h: ind_attr[ind] for h, ind in ind_dict.items()}
        else:
            ind_id = self.get_node_attributes('individual', node)
            return ind_attr[ind_id]

    def get_node_attributes(self, attr, node=None):

        if attr in self.attributes:
            return super().get_node_attributes(attr, node)
        elif attr in self.pedigree.attributes:
            return self.get_individual_attributes(attr, node)
        else:
            raise KeyError(f"Attribute {attr} is not defined.")

    def init_autosomal_graph(self):
        """
        Initialize the haplotype graph assuming autosomal inheritance pattern.
        """

        ind_idx = np.arange(1, self.pedigree.n_individuals + 1)
        paternal_idx = 2*ind_idx - 1
        maternal_idx = 2*ind_idx

        self.graph.add_nodes_from(paternal_idx)
        self.set_node_attributes(dict(zip(paternal_idx, self.pedigree.nodes)), 'individual')
        self.graph.add_nodes_from(maternal_idx)
        self.set_node_attributes(dict(zip(maternal_idx, self.pedigree.nodes)), 'individual')

        hap_dict = dict(zip(self.pedigree.nodes,
                            np.array([paternal_idx, maternal_idx]).T.tolist()))

        edges_to_add = []

        for n in hap_dict:

            for h_idx, parent in zip(hap_dict[n], self.pedigree.get_parents(n)):

                # If the parent is a node in the pedigree:
                if parent in hap_dict:
                    edges_to_add += [(p_h_idx, h_idx)
                                     for p_h_idx in hap_dict[parent]]

        self.graph.add_edges_from(edges_to_add)
        self.ind_to_hap = hap_dict

    def init_y_graph(self):
        """
        Initialize the haplotype graph assuming Chr Y inheritance pattern
        """

        assert "sex" in self.pedigree.attributes

        ind_sex = self.pedigree.get_node_attributes('sex')

        hap_dict = {}
        hap_counter = count(1)

        for n in self.pedigree.nodes:

            if ind_sex[n] == 1:
                hap_dict[n] = [next(hap_counter)]
                self.graph.add_node(hap_dict[n][0], individual=n)

        for n in hap_dict:

            father, mother = self.pedigree.get_parents(n)

            if father in hap_dict:
                self.graph.add_edge(hap_dict[father][0], hap_dict[n][0])

        self.ind_to_hap = hap_dict

    def init_mitochondrial_graph(self):
        """
        Initialize the haplotype graph assuming mitochondrial inheritance pattern
        """

        assert "sex" in self.pedigree.attributes

        hap_dict = {}

        for i, n in enumerate(self.pedigree.nodes, 1):
            hap_dict[n] = [i]
            self.graph.add_node(i, individual=n)

        for n in hap_dict:

            father, mother = self.pedigree.get_parents(n)

            if mother in hap_dict:
                self.graph.add_edge(hap_dict[mother][0], hap_dict[n][0])

        self.ind_to_hap = hap_dict

    def init_x_graph(self):

        assert "sex" in self.pedigree.attributes

        ind_sex = self.pedigree.get_node_attributes('sex')

        hap_dict = {}
        hap_counter = count(1)

        for n in self.pedigree.nodes:

            if ind_sex[n] == 1:
                hap_dict[n] = [next(hap_counter)]
            elif ind_sex[n] == 2:
                hap_dict[n] = [next(hap_counter), next(hap_counter)]

            for h_idx in hap_dict[n]:
                self.graph.add_node(h_idx, individual=n)

        for n in hap_dict:

            for h_idx, parent in zip(hap_dict[n], self.pedigree.get_parents(n)[::-1]):
                if parent in hap_dict:
                    self.graph.add_edges_from([(p_h_idx, h_idx)
                                               for p_h_idx in hap_dict[parent]])

        self.ind_to_hap = hap_dict

    def sample_path(self, haploid_probands=False):
        """
        Samples an inheritance path for the haplotypes of the proband
        and stores it in a Traversal object.

        :param haploid_probands: If `True` we assume that we sampled
        1 haplotype from each proband in the pedigree.

        :return: Traversal object
        """

        assert self.graph_type in ('autosomal', 'X')

        hap_time = self.get_node_attributes('time')

        # Create the traversal object:
        tr = Traversal(haploid_probands=haploid_probands)

        # Use this list to store nodes that we can sample ancestors for
        nodes_to_climb = []

        # Start from the probands in the pedigree:
        for proband in self.pedigree.probands():
            # For each proband, obtain their haplotypes
            proband_haps = self.ind_to_hap[proband]
            if haploid_probands:
                # If we're selecting 1 haplotype per proband, sample 1 randomly.
                proband_haps = rnd.choice(proband_haps, 1)

            # For each haplotype
            for h_idx in proband_haps:
                # Add it to the traversal graph:
                tr.graph.add_node(h_idx, time=hap_time[h_idx])
                # Add it to the list of nodes that we need to climb
                nodes_to_climb.append(h_idx)

        # While there are more nodes to climb
        while len(nodes_to_climb) > 0:

            # Obtain the node to process:
            curr_node, nodes_to_climb = nodes_to_climb[0], nodes_to_climb[1:]

            # If that node has been assigned a parent, continue
            if tr.predecessors(curr_node):
                continue

            # If not, choose its parent randomly from its list of predecessors:
            try:
                parent = rnd.choice(self.predecessors(curr_node))
            except ValueError:
                continue

            # if parent isn't already in the traversal object, add it with its time attribute:
            if parent not in tr.nodes:
                tr.graph.add_node(parent, time=hap_time[parent])

            tr.graph.add_edge(parent, curr_node)

            # If the parent has predecessors to climb to and hasn't been assigned a parent,
            # add to the list of nodes to climb:
            if len(self.predecessors(parent)) > 0 and len(tr.predecessors(parent)) == 0:
                nodes_to_climb.append(parent)

        tr.set_node_attributes(dict(zip(list(tr.nodes),
                                        list(tr.nodes))),
                               'haplotype')

        return tr

    def kinship_dok(self, individual_level=True, backward_only=False):
        """
        Implementation of kinship computation as a backward-forward message passing
        algorithm on the haplotype graph.
        """

        mtx = sparse.dok_matrix((self.n_individuals, self.n_individuals),
                                dtype=np.float32)

        n_idx = dict(zip(self.nodes, np.arange(self.n_individuals)))
        print(f"> inferring depth {datetime.datetime.now()}")
        nodes = self.infer_depth()
        print(f"> sorting nodes {datetime.datetime.now()}")
        sorted_nodes = sorted(nodes, key=nodes.get, reverse=True)

        print(f"> backward pass {datetime.datetime.now()}")
        # - - - - - Computing kinship - - - - -
        # Step 1: Backward pass (climbing up):
        for n in sorted_nodes:

            n_succ = self.successors(n)
            mtx[n_idx[n], n_idx[n]] = 1.

            for s in n_succ:
                mtx[n_idx[n]] += (1. / len(self.predecessors(s)))*mtx[n_idx[s]]

        if backward_only:
            return mtx

        print(f"> forward pass {datetime.datetime.now()}")
        # Step 2: Forward pass (descent):
        for n in sorted_nodes[::-1]:
            print('- - - >', n)
            n_pred = self.predecessors(n)

            if len(n_pred) > 0:

                print("before (Step 0):", mtx[n_idx[n]].toarray())
                for p in n_pred:
                    mtx[n_idx[n], n_idx[p]] = mtx[n_idx[p], n_idx[n]]
                nz = mtx[n_idx[n]].nonzero()
                print("before (Step 1):", mtx[n_idx[n]].toarray())
                for i, p in enumerate(n_pred):
                    #mtx[n_idx[n], n_idx[p]] = 0.
                    temp = mtx[n_idx[p]] * mtx[n_idx[p], n_idx[n]]
                    temp[nz] = 0.
                    mtx[n_idx[n]] += temp
                    #if i == 0:
                    #    mtx[n_idx[n]] = mtx[n_idx[p]] / len(n_pred)
                    #else:
                    #    mtx[n_idx[n]] += mtx[n_idx[p]] / len(n_pred)

            print("Node:", n, ": : :", mtx[n_idx[n]].toarray())

        # - - - - - Data Processing - - - - -
        if individual_level:

            ind_mtx = sparse.dok_matrix((self.pedigree.n_individuals,
                                         self.pedigree.n_individuals),
                                        dtype=np.float32)
            ped_nodes = list(self.pedigree.nodes)

            for idx_i, i in enumerate(ped_nodes):
                for idx_j, j in enumerate(ped_nodes[idx_i:], idx_i):

                    ind_mtx[idx_i, idx_j] = ind_mtx[idx_j, idx_i] = np.mean(
                        mtx[tuple(zip(*product([n_idx[h] for h in self.ind_to_hap[i]],
                                               [n_idx[h] for h in self.ind_to_hap[j]])))]
                    )

            return ind_mtx
        else:
            return mtx

    def kinship(self, individual_level=True, to_csr=True):
        """
        Implementation of kinship computation as a backward-forward message passing
        algorithm on the haplotype graph.
        """

        kin = {}
        print(f"> inferring depth {datetime.datetime.now()}")
        nodes = self.infer_depth()
        print(f"> sorting nodes {datetime.datetime.now()}")
        sorted_nodes = sorted(nodes, key=nodes.get, reverse=True)

        print(f"> backward pass {datetime.datetime.now()}")
        # - - - - - Computing kinship - - - - -
        # Step 1: Backward pass (climbing up):
        for n in sorted_nodes:

            n_succ = self.successors(n)

            kin[n] = {n: 1.}

            for s in n_succ:
                for k in kin[s]:
                    if k in kin[n]:
                        kin[n][k] += kin[s][k] / len(self.predecessors(s))
                    else:
                        kin[n][k] = kin[s][k] / len(self.predecessors(s))
                #kin[n][s] = 1. / len(self.predecessors(s))
                #kin[n].update({k: v * kin[n][s] for k, v in kin[s].items()})

        #return dict_to_csr(kin, self.nodes)

        print(f"> forward pass {datetime.datetime.now()}")
        # Step 2: Forward pass (descent):
        for n in sorted_nodes[::-1]:

            n_pred = self.predecessors(n)

            if len(n_pred) > 0:
                update_dict = {}
                for p in n_pred:
                    for k in set(kin[p]):
                        if k in n_pred:
                            update_dict[k] = 1. / len(n_pred)
                        elif k in update_dict:
                            update_dict[k] += kin[p][k] / len(n_pred)
                        else:
                            update_dict[k] = kin[p][k] / len(n_pred)

                print(f'Before ---> {n} <--->', kin[n])
                for k, v in update_dict.items():
                    kin[n][k] = v
                    if n in kin[k]:
                        kin[k][n] = v

                kin[n].update(update_dict)
                print(f'After ---> {n} <--->', kin[n])

        # - - - - - Data Processing - - - - -
        if individual_level:

            ped_nodes = list(self.pedigree.nodes)
            ind_kin = {n: {} for n in ped_nodes}

            for idx, i in enumerate(ped_nodes):
                for j in ped_nodes[idx:]:

                    if i not in self.ind_to_hap or j not in self.ind_to_hap:
                        ind_kin[i][j] = ind_kin[j][i] = 0.
                        continue

                    ks = []

                    for h_i, h_j in product(self.ind_to_hap[i], self.ind_to_hap[j]):

                        if h_j in kin[h_i]:
                            ks.append(kin[h_i][h_j])
                        else:
                            ks.append(0.)

                    ind_kin[i][j] = ind_kin[j][i] = np.mean(ks)

            if to_csr:
                return dict_to_csr(ind_kin, ped_nodes)
            else:
                return ind_kin

        else:
            if to_csr:
                return dict_to_csr(kin, self.nodes)
            else:
                return kin

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
