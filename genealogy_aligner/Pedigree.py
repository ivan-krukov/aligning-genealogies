import networkx as nx
import msprime as msp
from tqdm import tqdm
from itertools import count
from scipy.sparse import dok_matrix

import sys
import numpy as np
import pandas as pd
from numpy import random as rnd

from .Genealogical import Genealogical
from .Traversal import Traversal
from .Haplotype import Haplotype
from .Drawing import draw
from .utils import integer_dict


class Pedigree(Genealogical):
    """Handling Pedigree IO and calculations"""

    def __init__(self, graph=None):

        super().__init__(graph)
        self.haplotypes = None

    def add_couple(self, mat_id, pat_id, time):
        self.graph.add_node(mat_id, time=time)
        self.graph.add_node(pat_id, time=time)

    def add_individual(self, ind_id, time, sex=None):
        self.graph.add_node(ind_id, time=time, sex=sex)

    def add_child(self, child_id, mat_id, pat_id, time, sex=None):
        self.graph.add_node(child_id, time=time, sex=sex)
        self.graph.add_edge(mat_id, child_id)
        self.graph.add_edge(pat_id, child_id)

    def spouse(self, node):
        """
        For a given `node`, find its list of spouses in the pedigree
        """

        spouses = list(
            set(
                [
                    parent
                    for child in self.successors(node)
                    for parent in self.predecessors(child)
                    if parent != node
                ]
            )
        )

        if len(spouses) == 0:
            return [0]
        else:
            return spouses

    def kinship_lange(self, coefficient=2, progress=True):
        """Calculate the kinship matrix using the Lange kinship algorithm

        This algorithm uses the partial ordering from ``infer_depth()`` to assign indices in the output matrix
        
        Returns:
            np.array: dense symmetric matrix of kinsmip values
            dict: mapping from node labels to node index in matrix ``K``
        """
        G = self.graph
        label_gen = count(0)
        depth = self.infer_depth()
        ordered_label = sorted(G.nodes, key=lambda n: depth[n])
        ordered_index = [next(label_gen) for n in ordered_label]

        index_to_label = dict(zip(ordered_index, ordered_label))
        label_to_index = dict(zip(ordered_label, ordered_index))

        n = len(G.nodes)
        K = np.zeros((n, n), dtype=float)

        for node_idx in tqdm(range(n), disable=not progress):
            node = index_to_label[node_idx]
            if any(G.predecessors(node)):
                mother, father = G.predecessors(node)
                mat_idx, pat_idx = label_to_index[mother], label_to_index[father]
                K[node_idx, node_idx] = (1 + K[mat_idx, pat_idx]) / coefficient
            else:
                K[node_idx, node_idx] = 1 / coefficient

            for relative_idx in range(node_idx + 1, n):
                relative = index_to_label[relative_idx]
                if any(G.predecessors(relative)):
                    mother, father = G.predecessors(relative)
                    mat_idx, pat_idx = label_to_index[mother], label_to_index[father]
                    v = (K[node_idx, mat_idx] + K[node_idx, pat_idx]) / coefficient
                    K[node_idx, relative_idx] = v
                    K[relative_idx, node_idx] = v

        return K, label_to_index

    @staticmethod
    def infer_sex(ind_id, pat_id, mat_id):
        """Infer sex of an individual
        
        The inference is made based on whether the individual's ID is found in the ``maternal_id``
        or ``paternal_id`` column. Note that this can not infer the sex of probands.

        Args:
            ind_id (np.array(int)): IDs of individuals
            mat_id (np.array(int)): maternal IDs
            pat_id (np.array(int)): paternal IDs

        Returns:
            np.array: with same ordering as ``ind_id``, with ``1`` for males, ``2`` for females, ``-1`` for unknown
        """
        male = np.isin(ind_id, pat_id)
        female = np.isin(ind_id, mat_id)
        unknown = ~(male | female)
        sex = np.zeros_like(ind_id)
        sex[male] = 1
        sex[female] = 2
        # TODO: should this be `0`?
        sex[unknown] = -1
        return sex

    def get_parents(self, n):

        pred = self.predecessors(n)

        if len(pred) == 0:
            # no listed parents
            father, mother = (0, 0)
        elif len(pred) == 1:
            # 1 listed parent
            if "sex" in self.attributes:
                if self.get_node_attributes("sex", pred[0]) == 1:
                    father, mother = (pred[0], 0)
                elif self.get_node_attributes("sex", pred[0]) == 2:
                    father, mother = (0, pred[0])
                else:
                    # if sex of listed parent is unknown, choose randomly:
                    father, mother = rnd.choice([0, pred[0]], 2, replace=False)
            else:
                # if sex of listed parent is unknown, choose randomly:
                father, mother = rnd.choice([0, pred[0]], 2, replace=False)
        else:
            if "sex" in self.attributes:

                pred_sex = [self.get_node_attributes("sex", p) for p in pred]

                if pred_sex[0] == 1 or pred_sex[1] == 2:
                    father, mother = pred
                elif pred_sex[0] == 2 or pred_sex[1] == 1:
                    father, mother = pred[::-1]
                else:
                    # if sex of listed parents is unknown, choose randomly:
                    father, mother = rnd.choice(pred, 2, replace=False)
            else:
                # if sex of listed parents is unknown, choose randomly:
                father, mother = rnd.choice(pred, 2, replace=False)

        return father, mother

    @classmethod
    def from_balsac_table(cls, fname, sep="\t"):
        """Read BALSAC pedigree table

        This is an alias to :meth:`Pedigree.from_table`, with correct arguments in place

        The input table is expected to be a tab-separated file with the following columns:
        
        - `individual`: label for the individual
        - `father`: label of individual's father
        - `mother`: label of individual's mother
        - `sex`: sex of the individual: `1` male, `2` female, `-1` unknown

        Args:
           fname (path): path to the input table
           sep (str): separator between columns

        Returns:
           Pedigree: a pedigree with `depth`, and `sex` parameters assigned to each node
        """
        return Pedigree.from_table(
            fname, sep=sep, check_2_parents=False, attrs=["sex"], header=True
        )

    @classmethod
    def from_table(
        cls, f_name, attrs=("time",), header=False, check_2_parents=True, sep="\t"
    ):
        """Read pedigree from table
        
        The input table is required to have 3 following columns:
        
        - `individual`: label for the individual
        - `father`: label of individual's father
        - `mother`: label of individual's mother

        If `sex` column is present, `1` will be interpreted as male, `2` as female, `-1` as unknown.

        Any other additional columns, specified in ``attrs`` will be attached as attributes to the genealogy nodes.

        Example:
        
        To read in a BALSAC-type table, use::
            
            Pedigree.from_table('path/to/table', attrs=['sex'], header=True, check_2_parents=False)
        
        or::
           
            Pedigree.from_balsac_table('path/to-table')


        Args:
            f_name (path): path of the table to be read
            header (bool): is the header present in the input file?
            check_2_parents (bool): should we check that each individual has unique parents? 
            sep (str): Separator to use between columns. By default, an appropriate separator will be inferred using ``pandas.read_table``.

        Raises:
            ValueError: if `check_2_parents=True`, and ``father==mother`` for some row

        Returns:
            Pedigree: a pedigree, with ``time`` and ``sex`` parameters assigned to every node
        """

        ped = cls()
        required_columns = ["individual", "father", "mother"]

        if header:
            ped_df = pd.read_table(f_name, sep=sep)
            attrs = list(ped_df.columns[3:])
            ped_df.columns = required_columns + attrs
        else:
            ped_df = pd.read_table(
                f_name, sep=sep, names=required_columns + list(attrs)
            )

        # -------------------------------------------------------
        # Checking validity of table:

        if check_2_parents:
            cond = (
                ((ped_df["father"] == 0) & (ped_df["mother"] != 0))
                | ((ped_df["mother"] == 0) & (ped_df["father"] != 0))
                | ((ped_df["father"] != 0) & (ped_df["father"] == ped_df["mother"]))
            )

            if sum(cond) > 0:
                raise ValueError(
                    "Pedigree has individuals with 1 parent only. "
                    "Either set `check_2_parents` to False or remove those "
                    "individuals from the pedigree."
                )

        # -------------------------------------------------------
        # Add all nodes and edges to the graph:
        ped.graph.add_edges_from(zip(ped_df["father"], ped_df["individual"]))
        ped.graph.add_edges_from(zip(ped_df["mother"], ped_df["individual"]))

        # -------------------------------------------------------
        # Add node attributes:

        for attr in attrs:
            nx.set_node_attributes(
                ped.graph, dict(zip(ped_df["individual"], ped_df[attr])), attr
            )

        # Adding inferred time if not in attribute list:
        if "time" not in attrs:
            coal_time = ped.infer_depth(forward=False)
            nx.set_node_attributes(ped.graph, coal_time, "time")

        # Adding inferred sex if not in attribute list:
        if "sex" not in attrs:
            sex = ped.infer_sex(ped_df.individual, ped_df.father, ped_df.mother)
            nx.set_node_attributes(ped.graph, dict(zip(ped_df.individual, sex)), "sex")

        # -------------------------------------------------------
        # Final touches
        node_time = ped.get_node_attributes("time")
        ped.generations = max(node_time.values())

        if 0 in ped.graph:
            ped.graph.remove_node(0)

        return ped

    def to_table(self):
        """Convert a :class:`Pedigree` to a ``pandas.DataFrame``"""

        tbl = {
            node: {"individual": node, "mother": 0, "father": 0}
            for node in self.graph.nodes
        }

        for attr in self.attributes:
            attr_dict = self.get_node_attributes(attr)
            for node in self.graph.nodes:
                tbl[node][attr] = attr_dict[node]

        sex = self.get_node_attributes("sex")

        for node, parent in self.iter_edges(forward=False):

            if sex[parent] == 1:
                tbl[node]["father"] = parent
            elif sex[parent] == 2:
                tbl[node]["mother"] = parent
            else:  # sex unknown - implies sex == -1
                if tbl[node]["father"] == 0:
                    tbl[node]["father"] = parent
                elif tbl[node]["mother"] == 0:
                    tbl[node]["mother"] = parent
                else:
                    raise RuntimeError(f"Individual {node} has more than 2 parents")

        df = pd.DataFrame.from_dict(tbl, orient="index")
        df.sort_index(inplace=True)
        return df

    def to_msprime_pedigree(self):
        """Create an ``msprime`` representation of the pedigree

        This converts a :class:`Pedigree` into an ``msprime.Pedigree``

        Returns:
            msprime.Pedigree: an msprime Pedigree
        """

        tbl = self.to_table()

        individual = tbl.individual.values
        parent_IDs = np.vstack((tbl.father, tbl.mother)).transpose()
        parent_idx = msp.Pedigree.parent_ID_to_index(individual, parent_IDs)

        msp_ped = msp.Pedigree(individual, parent_idx, tbl.time.values)
        msp_ped.set_samples(sample_IDs=self.probands(), probands_only=True)

        return msp_ped

    def generate_msprime_simulations(
        self,
        Ne=1000,
        model_after="hudson",
        mu=None,
        length=None,
        rho=None,
        convert_to_traversal=True,
        random_seed=None,
    ):

        """Simulate with ``msprime`` upon a genealogy

        Todo:
            Always return an ``msprime.TreeSequence``, have an extra function to convert the output

        Args:
            Ne (int): effective population size to use after the pedigree simulation is complete
            model_after ['dtwf'/'hudson']: simulation model after the pedigree simulation is complete
            mu (float): mutation rate
            length (float): length of the genomic segment to simulate
            rho (float): recombination rate
            convert_to_traversal (bool): should the return type be a :class:`Traversal`?

            """

        if model_after:
            des = [msp.SimulationModelChange(self.generations, model_after)]
        else:
            des = []

        sim = msp.simulate(
            len(self.probands()),
            Ne=Ne,
            pedigree=self.to_msprime_pedigree(),
            model="wf_ped",
            length=length,
            mutation_rate=mu,
            recombination_rate=rho,
            demographic_events=des,
            random_seed=random_seed,
        )

        ts_nodes_to_ped_map = {}

        for n in sim.nodes():
            if n.individual != -1:
                ind_info = sim.individual(n.individual)
                ts_nodes_to_ped_map[n.id] = int(ind_info.metadata.decode())

        if convert_to_traversal:
            traversals = []
            for ts in sim.aslist():
                t = Traversal()
                t.graph.add_edges_from([(v, k) for k, v in ts.parent_dict.items()])
                t.ts_node_to_ped_node = {
                    k: v for k, v in ts_nodes_to_ped_map.items() if k in t.graph.nodes
                }
                # Set time information:
                nx.set_node_attributes(
                    t.graph, {n: ts.get_time(n) for n in t.nodes}, "time"
                )
                t.ploidy = 2
                traversals.append(t)
            return sim, traversals
        else:
            return sim, ts_nodes_to_ped_map

    @classmethod
    def simulate_from_founders(
        cls, n_founders, n_generations, avg_offspring=2, avg_immigrants=2
    ):
        """Simulate a genealogy forward in time, starting with `n_founders` individuals
        If an odd number of individuals are provide, an extra founder will be added

        Args:
            n_founders (int): number of founders starting the population
            n_generations (int): number of generations to simulate
            avg_offspring=2 (float): average number of children per family, mean of Poisson RV
            avg_immigrants=2 (float): average number of out-of family individuals added each generation

        Raises:
            RuntimeError: if a simulation is terminated before `n_generations`

        Returns:
            Pedigree: A wrapper around networkx.DiGraph of relationships.
        """

        ped = cls()
        ped.generations = n_generations
        id_counter = count(1)
        current_gen, next_gen = [], []

        for founder in range(n_founders):
            ind_id = next(id_counter)
            ped.graph.add_node(ind_id, time=n_generations)
            current_gen.append(ind_id)

        for t in range(n_generations - 1, -1, -1):

            # Add extra parent if necessary
            if len(current_gen) % 2 == 1:
                ind_id = next(id_counter)
                ped.graph.add_node(ind_id, time=t + 1)
                current_gen.append(ind_id)

            # Pick couples
            while len(current_gen):
                mat_id, pat_id = rnd.choice(current_gen, 2, replace=False)
                current_gen.remove(mat_id)
                current_gen.remove(pat_id)
                n_children = rnd.poisson(avg_offspring)

                for ch in range(n_children):
                    child_id = next(id_counter)
                    next_gen.append(child_id)
                    ped.add_child(child_id, mat_id, pat_id, time=t)

            # add extra out-of-family individuals - but not in the present
            if t > 1:
                n_immigrants = rnd.poisson(avg_immigrants)
                for ind in range(n_immigrants):
                    ind_id = next(id_counter)
                    next_gen.append(ind_id)
                    ped.add_individual(ind_id, t)

            if not next_gen:
                raise (
                    RuntimeError(
                        "Simulation terminated at time t="
                        + str(t)
                        + ", ("
                        + str(n_generations - t)
                        + " generations from founders)"
                    )
                )
            current_gen = next_gen
            next_gen = []

        if not nx.is_weakly_connected(ped.graph):
            # if multiple subgraphs, return largest
            largest = max(nx.weakly_connected_components(ped.graph), key=len)
            ped.graph = nx.subgraph(ped.graph, largest)

        return ped

    @classmethod
    def simulate_from_founders_with_sex(
        cls, n_founders, n_generations, avg_offspring=2, avg_immigrants=2, seed=None
    ):
        ped = cls()
        ped.generations = n_generations

        rng = rnd.default_rng(seed)
        current_males, current_females = [], []
        next_males, next_females = [], []

        id_counter = count(1)

        # assign sex to the founder generation
        for _ in range(n_founders):
            ind_id = next(id_counter)
            male = rng.random() < 0.5
            if male:
                current_males.append(ind_id)
                ped.graph.add_node(ind_id, time=n_generations, sex=1)
            else:
                current_females.append(ind_id)
                ped.graph.add_node(ind_id, time=n_generations, sex=2)

        for t in range(n_generations - 1, -1, -1):

            # pad the arrays if we have uneven sex ratio
            diff = len(current_males) - len(current_females)
            if diff > 0:
                for _ in range(diff):
                    ind_id = next(id_counter)
                    current_females.append(ind_id)
                    ped.graph.add_node(ind_id, time=t + 1, sex=2)
            elif diff < 0:
                for _ in range(-diff):
                    ind_id = next(id_counter)
                    current_males.append(ind_id)
                    ped.graph.add_node(ind_id, time=t + 1, sex=1)

            # Pick couples
            while len(current_males) and len(current_females):
                father = rng.choice(current_males)
                mother = rng.choice(current_females)
                current_males.remove(father)
                current_females.remove(mother)

                n_children = rng.poisson(avg_offspring)

                for ch in range(n_children):
                    child_id = next(id_counter)
                    child_male = rng.random() < 0.5
                    if child_male:
                        next_males.append(child_id)
                        ped.add_child(child_id, mother, father, time=t, sex=1)
                    else:
                        next_females.append(child_id)
                        ped.add_child(child_id, mother, father, time=t, sex=2)

            # add extra out-of-family individuals - but not in the present
            if t > 1:
                n_immigrants = rng.poisson(avg_immigrants)
                for _ in range(n_immigrants):
                    ind_id = next(id_counter)
                    ind_male = rng.random() < 0.5
                    if ind_male:
                        next_males.append(ind_id)
                        ped.add_individual(ind_id, t, sex=1)
                    else:
                        next_females.append(ind_id)
                        ped.add_individual(ind_id, t, sex=2)

            if not (next_males or next_females):
                raise (
                    RuntimeError(
                        "Simulation terminated at time t="
                        + str(t)
                        + ", ("
                        + str(n_generations - t)
                        + " generations from founders)"
                    )
                )
            current_males = next_males
            current_females = next_females
            next_males = []
            next_females = []

        if not nx.is_weakly_connected(ped.graph):
            # if multiple subgraphs, return largest
            largest = max(nx.weakly_connected_components(ped.graph), key=len)
            ped.graph = nx.subgraph(ped.graph, largest)

        return ped

    def get_haplotypes(self, ploidy=2):

        assert ploidy in (1, 2)

        self.haplotypes = {}

        for i, n in enumerate(self.nodes, 1):
            if ploidy == 2:
                self.haplotypes[n] = [Haplotype(2 * i - 1, n), Haplotype(2 * i, n)]
            else:
                self.haplotypes[n] = [Haplotype(i, n)]

        return self.haplotypes

    def sample_path(self, probands=None, ploidy=1):
        """Sample a coalescent path from a genealogy

        Starting at the probands (t=0), randomly choose a parent.
        Stop once founder individuals (t=max) are reached

        Returns:
            A `Traversal` object
        """

        if probands is None:
            probands = self.probands()

        tr = Traversal()
        tr.generations = self.generations

        # Obtain the haplotype data structure:
        hap_struct = self.get_haplotypes(ploidy)
        hap_to_ind = {}

        # For all the nodes in the pedigree:
        for i, n in enumerate(self.nodes, 1):

            # For all the haplotypes assigned to each node,
            # randomly pair them with the parents (without
            # replacement).
            for hap_obj, parent in zip(
                hap_struct[n], rnd.choice(self.get_parents(n), 2, replace=False)
            ):

                # If the parent is a node in the pedigree:
                if parent != 0:
                    # If the given haplotype hasn't been linked to a haplotype
                    # the parent, assign randomly from the parent's 2 copies:
                    if hap_obj.parent_haplotype is None:
                        hap_obj.parent_haplotype = rnd.choice(hap_struct[parent])

                    tr.graph.add_edge(hap_obj.parent_haplotype.id, hap_obj.id)

                hap_to_ind[hap_obj.id] = hap_obj.individual_id

        # Trim paths that end at non-proband haplotypes:
        non_proband_terminals = [
            n for n in tr.probands() if hap_to_ind[n] not in probands
        ]
        while len(non_proband_terminals) > 0:
            tr.graph.remove_nodes_from(non_proband_terminals)
            non_proband_terminals = [
                n for n in tr.probands() if hap_to_ind[n] not in probands
            ]

        # Assign the 'haplotype to individual' dictionary to the traversal object:
        hap_to_ind = {k: v for k, v in hap_to_ind.items() if k in tr.nodes}
        tr.ts_node_to_ped_node = hap_to_ind

        # Set time attribute information:
        node_time = self.get_node_attributes("time")
        nx.set_node_attributes(
            tr.graph, {h: node_time[i] for h, i in hap_to_ind.items()}, "time"
        )

        tr.ploidy = ploidy

        return tr

    def sample_haploid_path(self, seed, probands=None):

        rng = rnd.default_rng(seed)
        time = self.get_node_attributes("time")

        T = Traversal()
        T.generations = self.generations
        if probands is None:
            probands = self.probands()
        for proband in probands:
            T.graph.add_node(proband, time=time[proband])
        T.ts_node_to_ped_node = {}

        for t in range(self.generations):
            for node in T.nodes_at_generation(t):
                parents = list(self.graph.predecessors(node))
                if parents:
                    parent = rng.choice(parents)
                    T.graph.add_node(parent, time=time[parent])
                    T.graph.add_edge(parent, node)

        T.ploidy = 1
        return T

    def iter_trios(self):
        sex = self.get_node_attributes("sex")
        for node in self.iter_nodes():
            parents = self.parents(node)
            if parents:
                sex_dict = {sex[p]: p for p in parents}
                father, mother = sex_dict[1], sex_dict[2]
                yield {"father": father, "mother": mother, "child": node}

    def draw(self, **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""
        if "node_shape" in kwargs:
            return draw(self.graph, **kwargs)
        else:
            if "sex" in self.attributes:
                sex_to_shape = {1: "s", 2: "o", 3: "p"}
                return draw(
                    self.graph,
                    node_shape={
                        k: sex_to_shape[v]
                        for k, v in self.get_node_attributes("sex").items()
                    },
                    **kwargs,
                )
            else:
                return draw(self.graph, **kwargs)
