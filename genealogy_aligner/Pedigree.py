import networkx as nx
import msprime as msp
from itertools import count
import io
import numpy as np
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
    def from_msprime_pedigree(cls, fname, filter_zeros=True):

        import pandas as pd
        ped = cls()

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
        for f in range(families):
            mat_id, pat_id = 2*f, 2*f+1
            ped.add_couple(mat_id, pat_id, 0)
            current_gen.extend([mat_id, pat_id])

        id_counter = count(families*2)

        for t in range(1, generations+1):
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

        if t != generations:
            raise RuntimeError('Simulation terminated early')
        
        if not nx.is_weakly_connected(ped.graph):
            # if multiple subgraphs, return largest
            largest = max(nx.weakly_connected_components(ped.graph), key=len)
            Gl = nx.subgraph(ped.graph, largest)
            # relabel
            ped.graph = nx.convert_node_labels_to_integers(Gl, ordering='sorted')
        ped.generations = generations

        return ped

    def to_msprime_pedigree(self):

        txt = ""

        for n in self.nodes:
            p = self.predecessors(n)
            if p:
                txt += "\t".join(map(str, [n + 1, p[0] + 1, p[1] + 1])) + "\n"
            else:
                txt += "\t".join(map(str, [n + 1, 0, 0])) + "\n"

        return msp.Pedigree.read_txt(
            io.StringIO(txt)
        )

    def generate_msprime_simulations(self):

        return msp.simulate(len(self.probands()),
                            model='wf_ped',
                            pedigree=self.to_msprime_pedigree())

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
        T.graph.add_nodes_from(current_gen, time=self.generations)
        
        for t in reversed(range(self.generations)):
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
