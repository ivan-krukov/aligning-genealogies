import networkx as nx
import numpy as np
import copy
from scipy.sparse import dok_matrix
from collections import defaultdict
from itertools import count
import numpy.random as rnd
from tqdm import tqdm
import msprime
import tskit

from .Drawing import draw
from .Genealogical import Genealogical


class Traversal(Genealogical):

    def __init__(self, graph=None, haploid_probands=False):

        super().__init__(graph)
        self.haploid_probands = haploid_probands

    def to_coalescent_tree(self, add_common_ancestors=True, Ne=100):
        """
        Convert an inheritance path to a coalescent tree
        :param add_common_ancestors:
        :param Ne: The effective population size
        (Used to draw coalescent generation times for out-of-pedigree nodes)

        :return:
        """

        t_obj = copy.deepcopy(self)

        non_coalesc_nodes = t_obj.filter_nodes(
            lambda node, data: len(t_obj.successors(node)) == 1
        )

        edges_to_skipped_nodes = {}
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
                    edges_to_skipped_nodes[(pred_n, k)] = ped_nodes

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

                        edges_to_skipped_nodes[(ca_counter, n)] = ped_nodes

                    tree_founders = [f for f in tree_founders if f != n]
                tree_founders.append(ca_counter)

        t_obj.graph.remove_nodes_from(list(nx.isolates(t_obj.graph)))

        # Handling for out-of-pedigree nodes:

        out_ped = [n for n in t_obj.nodes if int(n) < 0]

        # Set the time attribute for out-of-pedigree nodes to a sample from geometric
        # distribution for now:
        self.set_node_attributes({
            n: np.random.geometric(1./Ne) + min([t_obj.get_node_attributes('time', s)
                                                 for s in t_obj.successors(n)])
            for n in out_ped
        }, 'time')

        t_obj.graph = nx.convert_node_labels_to_integers(
            t_obj.graph,
            first_label=1,
            label_attribute='haplotype'
        )

        # Set the haplotype to None for out-of-pedigree nodes:
        t_obj.set_node_attributes(
            dict(
                zip(out_ped, [None for _ in range(len(out_ped))])
            ),
            'haplotype'
        )

        t_obj.set_edge_attributes(edges_to_skipped_nodes, 'path')

        return t_obj

    def distances(self):
        """Calculate distances between nodes in the Traversal
        Returns:
            dict: ``distance[target][source]``"""
        dist = defaultdict(dict)
        for parent, child in self.trace_edges():
            deep = defaultdict(dict)
            for source, d in dist[parent].items():
                deep[child][source] = d + 1
            dist.update(deep)

            dist[child][parent] = 1
        return dist

    def distances_nx(self):
        """Calculate distances between nodes in the Traversal using ``networkx``
        Returns:
            dict: ``distance[source][target]``"""
        return dict(nx.all_pairs_shortest_path_length(nx.to_undirected(self.graph)))

    def distance_matrix(self, progress=False):
        dim = max(self.nodes) + 1
        D = dok_matrix((dim, dim))
        gen = nx.all_pairs_shortest_path_length(nx.to_undirected(self.graph))
        for source, table in tqdm(gen, total=dim, disable=not progress):
            for target, dist in table.items():
                D[source, target] = dist
        return D

    def parent_of(self, node):
        parents = list(self.graph.predecessors(node))
        return parents[0] if parents else None

    def to_coalescent(self):
        """Remove internal nodes from a Traversal
        Warning:
            This method is unstable
        """
        time = self.get_node_attributes("time")

        # dist = self.distances()
        dist = self.distances_nx()
        C = Traversal()
        C.generations = self.generations
        C.graph.add_nodes_from(self.probands(), time=0)

        for t in range(self.generations):
            for node in C.nodes_at_generation(t):
                parent = self.parent_of(node)
                while parent and self.graph.out_degree(parent) < 2:
                    parent = self.parent_of(parent)
                if parent:
                    C.graph.add_node(parent, time=time[parent])
                    C.graph.add_edge(parent, node, weight=dist[node][parent])

        C.graph.remove_nodes_from(list(nx.isolates(C.graph)))
        return C

    def to_tree_sequence(self, simplify=True):

        tables = msprime.TableCollection(1)

        coal_depth = self.get_node_attributes('time')
        msprime_id = {}
        for proband in self.probands():
            u = tables.nodes.add_row(
                time=coal_depth[proband], flags=msprime.NODE_IS_SAMPLE,
            )
            msprime_id[proband] = u
        founders = self.founders()

        visited_nodes = set()
        for node, parent in self.iter_edges(forward=False):
            if parent not in visited_nodes:
                t = coal_depth[parent]
                f = msprime.NODE_IS_SAMPLE if node in founders else 0
                u = tables.nodes.add_row(time=t, flags=f,)
                msprime_id[parent] = u
                visited_nodes.add(parent)
            else:
                u = msprime_id[parent]
            tables.edges.add_row(0, 1, u, msprime_id[node])

        tables.sort()
        if simplify:
            tables.simplify()
        # Unmark the initial generation as samples
        flags = tables.nodes.flags
        time = tables.nodes.time
        flags[:] = 0
        flags[time == 0] = msprime.NODE_IS_SAMPLE
        # The final tables must also have at least one population which
        # the samples are assigned to
        tables.populations.add_row()
        tables.nodes.set_columns(
            flags=flags, time=time, population=np.zeros_like(tables.nodes.population)
        )
        return tables.tree_sequence(), msprime_id

    def draw(self, **kwargs):
        if "reverse" not in kwargs:
            return draw(self.graph, reverse=True, **kwargs)
        else:
            return draw(self.graph, **kwargs)
