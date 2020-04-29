import networkx as nx
from itertools import count
import numpy as np
from numpy import random as rnd


class Genealogy(nx.DiGraph):
    def __init__(self, generations):
        super().__init__(self)
        self.generations = generations

    @property
    def n_individuals(self):
        return len(self.nodes)

    
    def add_couple(self, mat_id, pat_id, time):
        self.add_node(mat_id, time=time)
        self.add_node(pat_id, time=time)

        
    def add_individual(self, ind_id, time):
        self.add_node(ind_id, time=time)

        
    def add_child(self, child_id, mat_id, pat_id, time):
        self.add_node(child_id, time=time)
        self.add_edge(mat_id, child_id)
        self.add_edge(pat_id, child_id)


    def filter_nodes(self, predicate):
        node_list = []
        for node, data in self.nodes(data=True):
            if predicate(node, data):
                node_list.append(node)
        return node_list

    
    def founders(self):
        """Get a list of individuals at time 0"""
        return self.filter_nodes(
            lambda node, data: data['time'] == 0)
    

    def probands(self):
        """Get a list of individuals at present day"""
        return self.filter_nodes(
            lambda node, data: data['time'] == self.generations)

    
    def parents(self, i):
        return list(self.predecessors(i))


    def kinship(self):
        """Compute the kinship matrix for the genealogy"""
        n = self.n_individuals        
        K = np.zeros((n,n), dtype=float)
        
        for i in range(n):
            if any(self.predecessors(i)):
                p, q = self.predecessors(i)
                K[i, i] = 0.5 + (K[p,q]/2)
            else:
                K[i,i] = 0.5
            
            for j in range(i+1, n):
                if any(self.predecessors(j)):
                    p, q = self.predecessors(j)
                    K[i,j] = (K[i,p]/2) + (K[i,q]/2)
                    K[j,i] = K[i,j]
        return K

        

    @classmethod
    def from_founders(cls, families, generations, mean_offspring=2):
        """Simulate a genealogy forward in time, starting with `families` starting families

        Parameters
        ----------
        families: int
            Number of couples starting the population
        generations: int
            Number of generations to simulate
        mean_offspring: float
            Average number of children per family, mean of Poisson RV
        Returns
        -------
        nx.Graph
            networkx.Graph of relationships.
            Each node carries:
            time: int      - generation
            x: int         - index (out of N), for plotting
            parents: [int] - list of parent IDs - redundant - used for testing
        """
        G = cls(generations)
        

        current_gen = []
        next_gen = []
        # insert founder families
        for f in range(families):
            mat_id, pat_id = 2*f, 2*f+1
            G.add_couple(mat_id, pat_id, 0)
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
                    G.add_child(child_id, mat_id, pat_id, t)
                    
            current_gen = next_gen
            next_gen = []

        return G


    def draw(self, labels=True, ax=None):
        """Uses `graphviz` to plot the genealogy"""
        pos = nx.drawing.nx_agraph.graphviz_layout(self, prog='dot')
        nx.draw(self, pos=pos, with_labels=labels, node_shape='s', ax=ax)

