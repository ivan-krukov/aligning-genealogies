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
    def __init__(self, graph=None):
        super().__init__(graph)
        self.ploidy = None
        self.ts_node_to_ped_node = None
        self.ped_node_to_ts_edge = None

    def similarity(self, G):
        # A kinship-like distance function
        n = G.n_individuals
        K = np.zeros((n, n), dtype=float)

        for i in range(n):
            if i in self:
                K[i, i] = 0.5
                for j in range(i + 1, n):
                    if j in self:
                        if any(self.predecessors(j)):
                            p = next(self.predecessors(j))
                            K[i, j] = K[i, p] / 2
                            K[j, i] = K[i, j]
        return K

    def to_coalescent_tree(self, add_common_ancestors=True, inplace=False):

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

        # Set the time attribute for out-of-pedigree nodes to inf for now:
        nx.set_node_attributes(
            t_obj.graph, {ind: np.inf for ind in t_obj.nodes if int(ind) < 0}, "time"
        )

        t_obj.ts_node_to_ped_node = {
            k: v for k, v in self.ts_node_to_ped_node.items() if k in t_obj.nodes
        }

        t_obj.ped_node_to_ts_edge = {}

        for edge in edges_to_skipped_nodes:
            for n in edges_to_skipped_nodes[edge]:
                t_obj.ped_node_to_ts_edge[n] = edge

        if inplace:
            self = t_obj
        else:
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

    # TODO: remove the `kinship_like a`rgument
    def distance_matrix(self, progress=False, kinship_like=False, kinship_coeff=2.0):
        dim = max(self.nodes) + 1
        D = dok_matrix((dim, dim))
        # TODO: Is this the best way to get tree lengths?
        gen = nx.all_pairs_dijkstra_path_length(nx.to_undirected(self.graph))
        for source, table in tqdm(gen, total=dim, disable=not progress):
            for target, dist in table.items():
                if kinship_like:
                    dist = kinship_coeff ** (-dist)
                D[source, target] = dist
        return D

    def first_parent_of(self, node):
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
                parent = self.first_parent_of(node)
                while parent and self.graph.out_degree(parent) < 2:
                    parent = self.first_parent_of(parent)
                if parent:
                    C.graph.add_node(parent, time=time[parent])
                    C.graph.add_edge(parent, node, weight=dist[node][parent])

        C.graph.remove_nodes_from(list(nx.isolates(C.graph)))
        return C

    @classmethod
    def from_tree_sequence(cls, ts, labels=None):
        """WARNING: only works for single-tree TS!"""
        # use provided label mapping and add extra labels for out-of-tree
        def labels_for_tree(nodes_table, labels):
            L = len(nodes_table)
            oot_counter = count(max(labels.values()) + 1)
            assigned_labels = {}
            for i in range(L):
                assigned_labels[i] = (
                    int(labels[i]) if i in labels else next(oot_counter)
                )
            return assigned_labels

        tables = ts.dump_tables()

        if labels is not None:
            labels = labels_for_tree(tables.nodes, labels)
        else:
            # UGLY!
            labels = {i: i for i in range(len(tables.nodes))}

        T = Traversal()
        time = tables.nodes.time
        for node, t in enumerate(time):
            T.graph.add_node(labels[node], time=t)

        for child, parent in zip(tables.edges.child, tables.edges.parent):
            T.graph.add_edge(
                labels[parent], labels[child], weight=time[parent] - time[child]
            )

        T.generations = np.max(time)
        T.ploidy = 1  # ???
        return T

    def to_tree_sequence(self, simplify=True):

        tables = msprime.TableCollection(1)

        coal_depth = self.get_node_attributes("time")
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
            mapping = tables.simplify()
            simplified_msprime_id = {}
            for i, node in enumerate(msprime_id.keys()):
                simplified_msprime_id[node] = mapping[i]
                msprime_id = simplified_msprime_id
        tables.populations.add_row()
        tables.nodes.set_columns(
            flags=tables.nodes.flags,
            time=tables.nodes.time,
            population=np.zeros_like(tables.nodes.population),
        )
        return tables.tree_sequence(), msprime_id

    def draw(self, **kwargs):
        if "reverse" not in kwargs:
            return draw(self.graph, reverse=True, **kwargs)
        else:
            return draw(self.graph, **kwargs)
