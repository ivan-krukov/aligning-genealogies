import networkx as nx
import numpy as np
from numpy import random as rnd

# the format of the genealogy is:
# ind_id \t father_id \t mother_id \t time (in generations, optional)

def simulate_wf_genealogy(sample_size, population_size, time, individual_completeness = 1):
    """Simulate a genealogy from a Wright-Fisher population
       
    Parameters
    ----------
    sample_size: int
        Size of the current generation
    population_size: int
        Size of the Wright-Fisher population
    time: int
        Number of generations to simulate
    individual_completeness: float
        Probability that parents of any given individual are in the genealogy
    Returns
    -------
    nx.Graph
        networkx.Graph of relationships. Each node carries a time attribute
    """
    
    G = nx.Graph() # should this be directed?

    current_gen = list(range(sample_size))
    G.add_nodes_from(current_gen, time=0)
    next_gen = []

    idx = 0
    for t in range(1, time + 1):
        idx += len(current_gen)
        for individual in current_gen:
            if rnd.uniform(0,1) < p:
                parents = (idx + 1, idx + 2)
                idx += 2
                next_gen.extend(parents)
                G.add_nodes_from(parents, time=t)
                G.add_edge(individual, parents[0])
                G.add_edge(individual, parents[1])
        current_gen = next_gen
        next_gen = []

        
    return G
        
    
    
