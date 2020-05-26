import networkx as nx
import msprime as msp
from itertools import count

import numpy as np
import pandas as pd
from numpy import random as rnd
import matplotlib.pyplot as plt

from .Genealogical import Genealogical
from .Traversal import Traversal


class Pedigree(Genealogical):
    
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

    @classmethod
    def read_balsac(cls, fname):
        balsac_columns = ['ind', 'father', 'mother', 'sex']
        ped = cls()
        ped_df = pd.read_csv(fname, sep="\t")
        if ped_df.columns.tolist() != balsac_columns:
            raise RuntimeError("Unexpected column headers - required format" + str(balsac_columns))
        
        time = Pedigree.infer_time(ped_df['ind'], ped_df['mother'], ped_df['father'])
        ped.generations = max(time)

        for i, row in ped_df.iterrows():
            ind, pat_id, mat_id, sex = row
            if pat_id == mat_id == 0:
                ped.add_individual(ind, time[ind])
            else:
                ped.add_child(ind, pat_id, mat_id, time[ind])

        return ped

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
    
    def sample_path(self, probands=None, ploidy=1):
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

        node_time = self.get_node_attributes('time')

        ind_to_hap = {}

        for i, n in enumerate(self.nodes, 1):

            if ploidy == 1:
                hap_id = [i]
            else:
                hap_id = [2 * i - 1, 2 * i]

            ind_to_hap[n] = dict(
                zip(
                    hap_id,
                    rnd.choice(self.get_parents(n), 2, replace=False)
                )
            )

        hap_to_ind = {hap: ind for ind, di in ind_to_hap.items() for hap in di.keys()}

        for pid in probands:
            for hap_id, parent_id in ind_to_hap[pid].items():

                current_hap_id = hap_id
                inherited_from = parent_id  # bequeathing parent

                parent_hap_id = rnd.choice(list(ind_to_hap[inherited_from].keys()))

                # Once you link one of the parent's haplotypes to the child's, remove
                # the other from the dictionary (this ensures no diverging paths going
                # backward in time):
                ind_to_hap[inherited_from] = {k: v for k, v in ind_to_hap[inherited_from].items()
                                              if k == parent_hap_id}

                while True:
                    tr.graph.add_edge(parent_hap_id, current_hap_id)

                    current_hap_id = parent_hap_id
                    inherited_from = ind_to_hap[inherited_from][current_hap_id]

                    if inherited_from == 0:
                        break
                    else:
                        parent_hap_id = rnd.choice(list(ind_to_hap[inherited_from].keys()))
                        ind_to_hap[inherited_from] = {k: v for k, v in ind_to_hap[inherited_from].items()
                                                      if k == parent_hap_id}

        hap_to_ind = {k: v for k, v in hap_to_ind.items() if k in tr.nodes}
        tr.ts_node_to_ped_node = hap_to_ind

        # Set time attribute information:
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
