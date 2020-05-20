import networkx as nx
import msprime as msp
from itertools import count
import io
import numpy as np
import pandas as pd
from numpy import random as rnd
from Genealogical import Genealogical
from Traversal import Traversal


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

    def kinship(self):
        """Compute the kinship matrix for the genealogy"""
        n = self.n_individuals        
        K = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            if any(self.predecessors(i)):
                p, q = self.predecessors(i)
                K[i, i] = 0.5 + (K[p,q]/2)
            else:
                K[i, i] = 0.5
            
            for j in range(i+1, n):
                if any(self.predecessors(j)):
                    p, q = self.predecessors(j)
                    K[i,j] = (K[i,p]/2) + (K[i,q]/2)
                    K[j,i] = K[i,j]
        return K

    @classmethod
    def from_msprime_pedigree(cls, fname, header=False, filter_zeros=True):

        ped = cls()

        if header:
            gen_df = pd.read_csv(fname, sep="\t")
        else:
            gen_df = pd.read_csv(fname, sep="\t",
                                 names=["ind_id", "father", "mother", "time"])

        ped.generations = gen_df['time'].max()

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

        if filter_zeros:
            ped.graph.remove_node(0)

        return ped

    @classmethod
    def from_founders(cls, families, generations, mean_offspring=2, mean_out_of_family=2):
        """Simulate a genealogy forward in time, starting with `families` starting families

        Parameters
        ----------
        families: int
            Number of couples starting the population
        generations: int
            Number of generations to simulate
        mean_offspring: float
            Average number of children per family, mean of Poisson RV
        mean_out_of_family: float
            Average number of out-of family individuals added each generation
        Returns
        -------
        nx.Graph
            networkx.Graph of relationships.
            Each node carries:
            time: int      - generation
            x: int         - index (out of N), for plotting
            parents: [int] - list of parent IDs - redundant - used for testing
        """
        ped = cls()

        current_gen = []
        next_gen = []
        # insert founder families
        for f in range(1, 2*(families + 1), 2):
            mat_id, pat_id = f, f+1
            ped.add_couple(mat_id, pat_id, generations)
            current_gen.extend([mat_id, pat_id])

        id_counter = count(1 + 2*families)

        for t in range(generations - 1, -1, -1):
            # if one individual is left, it produces no offspring
            while len(current_gen) >= 2:
                mat_id, pat_id = rnd.choice(current_gen, 2, replace=False)
                current_gen.remove(mat_id)
                current_gen.remove(pat_id)
                children = rnd.poisson(mean_offspring)

                for ch in range(children):
                    child_id = next(id_counter)
                    next_gen.append(child_id)
                    ped.add_child(child_id, mat_id, pat_id, t)

            # add extra out-of-family individuals
            new_arrivals = rnd.poisson(mean_out_of_family)
            for ind in range(new_arrivals):
                ind_id = next(id_counter)
                next_gen.append(ind_id)
                ped.add_individual(ind_id, t)

            current_gen = next_gen
            next_gen = []

        if t != 0:
            raise RuntimeError('Simulation terminated early')
        
        if not nx.is_weakly_connected(ped.graph):
            # if multiple subgraphs, return largest
            largest = max(nx.weakly_connected_components(ped.graph), key=len)
            Gl = nx.subgraph(ped.graph, largest)
            # relabel
            ped.graph = nx.convert_node_labels_to_integers(Gl,
                                                           ordering='sorted',
                                                           first_label=1)

        ped.generations = generations

        return ped

    def to_msprime_pedigree(self, f_name=None, header=False):

        ped_df = []
        node_time = self.get_node_attributes('time')

        for n in self.nodes:
            p = self.predecessors(n)
            if len(p) == 2:
                ped_df.append([n, p[0], p[1], node_time[n]])
            elif len(p) == 1:
                ped_df.append([n, p[0], 0, node_time[n]])
            else:
                ped_df.append([n, 0, 0, node_time[n]])

        ped_df = pd.DataFrame(ped_df,
                              columns=["ind_id", "father", "mother", "time"])

        if f_name is None:

            sio = io.StringIO()
            ped_df.to_csv(sio, sep="\t", index=False)
            sio.seek(0)
            msp_ped = msp.Pedigree.read_txt(sio, time_col=3)

            msp_ped.set_samples(sample_IDs=self.probands())

            return msp_ped
        else:
            ped_df.to_csv(f_name, index=False, header=header, sep="\t")

    def generate_msprime_simulations(self,
                                     Ne=100,
                                     model='wf_ped',
                                     mu=1e-8,
                                     length=1e6,
                                     rho=1e-8):

        rm = msp.RecombinationMap(
            [0, int(length)],
            [rho, 0],
            discrete=True
        )

        des = [
            msp.SimulationModelChange(self.generations, model)
        ]

        return msp.simulate(len(self.probands()),
                            Ne=Ne,
                            pedigree=self.to_msprime_pedigree(),
                            model=model,
                            mutation_rate=mu,
                            recombination_map=rm,
                            end_time=self.generations,
                            demographic_events=des)

    def sample_path(self):
        """
        Sample a coalescent path from a genealogy

        Starting at the probands, randomly choose a parent.
        Stop once founder individuals (t=0) are reached

        Returns:
        --------
        A `Traversal` object
        """

        current_gen = set(self.probands())
        T = Traversal()
        T.generations = self.generations
        T.graph.add_nodes_from(current_gen, time=0)
        
        for t in range(1, self.generations + 1):
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
