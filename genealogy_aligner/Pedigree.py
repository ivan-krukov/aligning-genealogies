import networkx as nx
import msprime as msp
from itertools import count

import numpy as np
import pandas as pd
from numpy import random as rnd
import matplotlib.pyplot as plt

from .Genealogical import Genealogical
from .Traversal import Traversal
from .Haplotype import Haplotype


class Pedigree(Genealogical):

    def __init__(self, graph=None):
        super().__init__(graph)
        self.haplotypes = None
    
    def add_couple(self, mat_id, pat_id, time):
        self.graph.add_node(mat_id, time=time)
        self.graph.add_node(pat_id, time=time)

    def add_individual(self, ind_id, time):
        self.graph.add_node(ind_id, time=time)

    def add_child(self, child_id, mat_id, pat_id, time):
        self.graph.add_node(child_id, time=time)
        self.graph.add_edge(mat_id, child_id)
        self.graph.add_edge(pat_id, child_id)

    @staticmethod
    def infer_time(ind_id, pat_id, mat_id):
        n = len(ind_id)
        assert n == len(mat_id)
        assert n == len(pat_id)

        depth = np.zeros(n, dtype=int)
        if n == 1:
            return depth

        parents = ind_id[(mat_id == 0) & (pat_id == 0)]

        for i in np.arange(1, n+1):
            # all the individuals whose parents are founders
            children = np.isin(mat_id, parents) | np.isin(pat_id, parents)
            if i == n:
                raise RuntimeError("Impossible pedigree - someone is their own ancestor")
            if np.any(children):
                depth[children] = i
                parents = ind_id[children]
            else:
                break

        # flip!
        return dict(zip(ind_id, np.abs(depth - max(depth))))

    def get_parents(self, n):

        pred = self.predecessors(n)

        if len(pred) == 0:
            # no listed parents
            father, mother = (0, 0)
        elif len(pred) == 1:
            # 1 listed parent
            if 'sex' in self.attributes:
                if self.get_node_attributes('sex', pred[0]) == 1:
                    father, mother = (pred[0], 0)
                elif self.get_node_attributes('sex', pred[0]) == 2:
                    father, mother = (0, pred[0])
                else:
                    # if sex of listed parent is unknown, choose randomly:
                    father, mother = rnd.choice([0, pred[0]], 2, replace=False)
            else:
                # if sex of listed parent is unknown, choose randomly:
                father, mother = rnd.choice([0, pred[0]], 2, replace=False)
        else:
            if 'sex' in self.attributes:

                pred_sex = [self.get_node_attributes('sex', p) for p in pred]

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
    def from_table(cls, f_name, attrs=('time',),
                   header=False, check_2_parents=True):

        ped = cls()

        if header:
            ped_df = pd.read_csv(f_name, sep="\t")
            attrs = list(ped_df.columns[3:])
            ped_df.columns = ["individual", "father", "mother"] + attrs
        else:
            ped_df = pd.read_csv(f_name, sep="\t",
                                 names=["individual", "father", "mother"] + list(attrs))

        # -------------------------------------------------------
        # Checking validity of table:

        if check_2_parents:
            cond = (
                    ((ped_df['father'] == 0) & (ped_df['mother'] != 0)) |
                    ((ped_df['mother'] == 0) & (ped_df['father'] != 0)) |
                    ((ped_df['father'] != 0) & (ped_df['father'] == ped_df['mother']))
            )

            if sum(cond) > 0:
                raise ValueError("Pedigree has individuals with 1 parent only. "
                                 "Either set `check_2_parents` to False or remove those "
                                 "individuals from the pedigree.")

        # -------------------------------------------------------
        # Add all nodes and edges to the graph:
        ped.graph.add_edges_from(zip(ped_df["father"], ped_df["individual"]))
        ped.graph.add_edges_from(zip(ped_df["mother"], ped_df["individual"]))

        # -------------------------------------------------------
        # Add node attributes:

        for attr in attrs:
            nx.set_node_attributes(ped.graph,
                                   dict(zip(ped_df['individual'], ped_df[attr])),
                                   attr)

        # Adding inferred time if not in attribute list:
        if 'time' not in attrs:
            nx.set_node_attributes(ped.graph,
                                   Pedigree.infer_time(ped_df['individual'],
                                                       ped_df['father'],
                                                       ped_df['mother']),
                                   'time')

        # Adding inferred sex if not in attribute list:
        if 'sex' not in attrs:

            def infer_sex(node_id):
                if node_id in ped_df['father']:
                    return 1
                elif node_id in ped_df['mother']:
                    return 2
                else:
                    return -1

            ped_df['sex'] = ped_df['ind_id'].apply(infer_sex)

            nx.set_node_attributes(ped.graph,
                                   dict(zip(ped_df['ind_id'], ped_df['sex'])),
                                   'sex')

        # -------------------------------------------------------
        # Final touches

        node_time = ped.get_node_attributes('time')
        ped.generations = max(node_time.values())

        ped.graph.remove_node(0)

        return ped

    def to_table(self, f_name=None, header=False, attrs=None):

        if attrs is None:
            attrs = self.attributes

        ped_df = []
        
        for i, n in enumerate(self.nodes):

            node_attr = [self.get_node_attributes(at, n) for at in attrs]
            ped_df.append([n] + list(self.get_parents(n)) + node_attr)

        ped_df = pd.DataFrame(ped_df,
                              columns=['individual', 'father', 'mother'] + list(attrs))

        if f_name is None:
            return ped_df
        else:
            ped_df.to_csv(f_name, sep="\t", header=header, index=False)

    def to_msprime_pedigree(self):

        tbl = self.to_table()

        individual = tbl.individual.values
        parent_IDs = np.vstack((tbl.father, tbl.mother)).transpose()
        parent_idx = msp.Pedigree.parent_ID_to_index(individual, parent_IDs)
        
        msp_ped = msp.Pedigree(individual, parent_idx, tbl.time.values)
        msp_ped.set_samples(sample_IDs=self.probands(), probands_only=True)

        return msp_ped

    def generate_msprime_simulations(self,
                                     Ne=100,
                                     model_after='dtwf',
                                     mu=1e-8,
                                     length=1e6,
                                     rho=1e-8,
                                     convert_to_traversal=True):

        rm = msp.RecombinationMap(
            [0, int(length)],
            [rho, 0],
            discrete=True
        )

        if model_after:
            des = [
                msp.SimulationModelChange(self.generations, model_after)
            ]
        else:
            des = []

        sim = msp.simulate(len(self.probands()),
                           Ne=Ne,
                           pedigree=self.to_msprime_pedigree(),
                           model='wf_ped',
                           mutation_rate=mu,
                           recombination_map=rm,
                           demographic_events=des)

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
                t.ts_node_to_ped_node = {k: v for k, v in ts_nodes_to_ped_map.items()
                                         if k in t.graph.nodes}
                # Set time information:
                nx.set_node_attributes(t.graph,
                                       {n: ts.get_time(n) for n in t.nodes},
                                       'time')
                traversals.append(t)
            return sim, traversals
        else:
            return sim, ts_nodes_to_ped_map

    @classmethod
    def simulate_from_founders(cls, n_founders, n_generations, avg_offspring=2, avg_immigrants=2):
        """Simulate a genealogy forward in time, starting with `n_founders` individuals
        If an odd number of individuals are provide, an extra founder will be added

        Parameters
        ----------
        n_founders: int
            Number of founders starting the population
        n_generations: int
            Number of generations to simulate
        avg_offspring=2: float
            Average number of children per family, mean of Poisson RV
        avg_immigrants=2: float
            Average number of out-of family individuals added each generation
        Raises
        ------
        RuntimeError
            If a simulation is terminated before `n_generations`
        Returns
        -------
        Pedigree
            A wrapper around networkx.DiGraph of relationships.
            Each node carries following attributes:
            time: int      - generation back in time
            Founders all have time=`n_generations`
        """

        ped = cls()
        ped.generations = n_generations
        id_counter = count(1)
        current_gen, next_gen = [], []

        for founder in range(n_founders):
            ind_id = next(id_counter)
            ped.graph.add_node(ind_id, time=n_generations)
            current_gen.append(ind_id)

        for t in range(n_generations-1, -1, -1):

            # Add extra parent if necessary
            if len(current_gen) % 2 == 1:
                ind_id = next(id_counter)
                ped.graph.add_node(ind_id, time=t+1)
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
                raise(RuntimeError('Simulation terminated at time t=' + str(t) +
                                   ', (' + str(n_generations-t) +
                                   ' generations from founders)'))
            current_gen = next_gen
            next_gen = []

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
                self.haplotypes[n] = [
                    Haplotype(2 * i - 1, n),
                    Haplotype(2 * i, n)
                ]
            else:
                self.haplotypes[n] = [
                    Haplotype(i, n)
                ]

        return self.haplotypes

    def sample_path(self, probands=None, ploidy=2):
        """
        Sample a coalescent path from a genealogy

        Starting at the probands (t=0), randomly choose a parent.
        Stop once founder individuals (t=max) are reached

        Returns:
        --------
        A `Traversal` object
        """

        assert ploidy in (1, 2)

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
                    hap_struct[n],
                    rnd.choice(self.get_parents(n), 2, replace=False)):

                # If the parent is a node in the pedigree:
                if parent != 0:
                    # If the given haplotype hasn't been linked to a haplotype
                    # the parent, assign randomly from the parent's 2 copies:
                    if hap_obj.parent_haplotype is None:
                        hap_obj.parent_haplotype = rnd.choice(hap_struct[parent])

                    tr.graph.add_edge(hap_obj.parent_haplotype.id, hap_obj.id)

                hap_to_ind[hap_obj.id] = hap_obj.individual_id

        # Trim paths that end at non-proband haplotypes:
        non_proband_terminals = [n for n in tr.probands(use_time=False)
                                 if hap_to_ind[n] not in probands]
        while len(non_proband_terminals) > 0:
            tr.graph.remove_nodes_from(non_proband_terminals)
            non_proband_terminals = [n for n in tr.probands(use_time=False)
                                     if hap_to_ind[n] not in probands]

        # Assign the 'haplotype to individual' dictionary to the traversal object:
        hap_to_ind = {k: v for k, v in hap_to_ind.items() if k in tr.nodes}
        tr.ts_node_to_ped_node = hap_to_ind

        # Set time attribute information:
        node_time = self.get_node_attributes('time')
        nx.set_node_attributes(tr.graph,
                               {h: node_time[i] for h, i in hap_to_ind.items()},
                               'time')

        return tr

    def draw(self, ax=None, figsize=(8, 6), node_color=None, labels=True,
             default_color='#2b8cbe', **kwargs):
        """Uses `graphviz` `dot` to plot the genealogy"""

        if 'sex' not in self.attributes:
            super().draw(labels=labels, node_color=node_color,
                         default_color=default_color, ax=ax, **kwargs)
        else:

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)

            if node_color is None:
                node_col = dict(zip(self.nodes, [default_color] * self.n_individuals))
            else:
                node_col = node_color
                for n in self.nodes:
                    if n not in node_col:
                        node_col[n] = default_color

            pos = self.get_graphviz_layout()

            node_sex = self.get_node_attributes('sex')
            males = [n for n, s in node_sex.items() if s == 1]
            females = [n for n, s in node_sex.items() if s == 2]
            unknown = [n for n, s in node_sex.items() if s == -1]

            # Draw nodes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=females,
                                   node_shape='o',
                                   node_color=[node_col[f] for f in females],
                                   ax=ax, **kwargs)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=males,
                                   node_shape='s',
                                   node_color=[node_col[m] for m in males],
                                   ax=ax, **kwargs)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=unknown,
                                   node_shape='p',
                                   node_color=[node_col[u] for u in unknown],
                                   ax=ax, **kwargs)

            # Draw labels
            if labels:
                nx.draw_networkx_labels(self.graph, pos, font_color='white',
                                        font_size=8, ax=ax)

            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, ax=ax, **kwargs)

            # Turn off axis
            ax.set_axis_off()
