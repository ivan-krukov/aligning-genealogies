import networkx as nx
import msprime as msp
from itertools import count

import numpy as np
import pandas as pd
from numpy import random as rnd

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

    def parents(self, node):
        return self.predecessors(node)


    @classmethod
    def from_msprime_pedigree(cls, fname, header=False, filter_zeros=True):

        ped = cls()

        if header:
            gen_df = pd.read_csv(fname, sep="\t")
        else:
            gen_df = pd.read_csv(fname, sep="\t",
                                 names=["ind_id", "father", "mother", "time"])

        ped.generations = gen_df['time'].max()

        # Filter out entries with node 0:
        if filter_zeros:
            gen_df = gen_df[(gen_df.iloc[:, :-1] != 0).all(axis=1)]

        # -------------------------------------------------------
        # Add all nodes and edges to the graph:
        ped.graph.add_edges_from(zip(gen_df["father"], gen_df["ind_id"]))
        ped.graph.add_edges_from(zip(gen_df["mother"], gen_df["ind_id"]))

        # -------------------------------------------------------
        # Add node attributes:
        nx.set_node_attributes(ped.graph, dict(zip(gen_df['ind_id'], gen_df['time'])), 'time')

        def infer_sex(node_id):
            if node_id in gen_df['father']:
                return 'M'
            elif node_id in gen_df['mother']:
                return 'F'
            else:
                return 'U'

        gen_df['sex'] = gen_df['ind_id'].apply(infer_sex)

        nx.set_node_attributes(ped.graph, dict(zip(gen_df['ind_id'], gen_df['sex'])), 'sex')

        return ped

    def to_table(self):
        individual = np.array(self.graph.nodes)
        time = self.attribute_array(individual, 'time')
        parents = np.zeros((len(individual), 2))
        father = np.zeros_like(individual)
        mother = np.zeros_like(individual)
        
        for i, node in enumerate(self.graph.nodes):
            pred = list(self.graph.predecessors(node))
            if pred:
                father[i] = pred[0]
                mother[i] = pred[1]
        tbl = np.vstack((individual, father, mother, time)).transpose()

        return pd.DataFrame(tbl, columns=['individual', 'father', 'mother', 'time'])
                        

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
                                     rho=1e-8):

        rm = msp.RecombinationMap(
            [0, int(length)],
            [rho, 0],
            discrete=True
        )

        des = [
            msp.SimulationModelChange(self.generations, model_after)
        ]

        return msp.simulate(len(self.probands()),
                            Ne=Ne,
                            pedigree=self.to_msprime_pedigree(),
                            model='wf_ped',
                            mutation_rate=mu,
                            recombination_map=rm,
                            demographic_events=des)

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
                raise(RuntimeError('Simulation terminated at time t=' + str(t) + ', (' + str(n_generations-t) + ' generations from founders)'))
            current_gen = next_gen
            next_gen = []

        if not nx.is_weakly_connected(ped.graph):
            # if multiple subgraphs, return largest
            largest = max(nx.weakly_connected_components(ped.graph), key=len)
            ped.graph = nx.subgraph(ped.graph, largest)

        return ped

    
    def sample_path(self):
        """
        Sample a coalescent path from a genealogy

        Starting at the probands (t=0), randomly choose a parent.
        Stop once founder individuals (t=max) are reached

        Returns:
        --------
        A `Traversal` object
        """

        current_gen = set(self.probands())
        T = Traversal()
        T.generations = self.generations
        T.graph.add_nodes_from(current_gen, time=self.generations)
        
        for t in range(self.generations):
            prev_gen = set()
            for individual in current_gen:
                parents = self.predecessors(individual)
                if parents:
                    parent = rnd.choice(parents)
                    T.graph.add_node(parent, time=t)
                    T.graph.add_edge(parent, individual)
                    prev_gen.add(parent)

            current_gen = prev_gen
        return T
