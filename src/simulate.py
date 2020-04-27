import networkx as nx
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt


def simulate_wf_genealogy(sample_size, population_size, generations, individual_completeness = 1):
    """Simulate a genealogy from a Wright-Fisher population
       
    Parameters
    ----------
    sample_size: int
        Size of the current generation
    population_size: int
        Size of the Wright-Fisher population
    generations: int
        Number of generations to simulate
    individual_completeness: float
        Probability that parents of any given individual are in the genealogy
    Returns
    -------
    nx.Graph
        networkx.Graph of relationships.
        Each node carries:
        time: int      - generation
        x: int         - index (out of N), for plotting
        parents: [int] - list of parent IDs - redundant - used for testing
    """
    
    G = nx.Graph() # should this be directed?

    # Node indexing is 1-based
    current_gen = set(range(1, sample_size+1))
    for i in current_gen:
        G.add_node(i, generation=1, x=i, parents=[])
    
    next_gen = set()

    for t in range(2, generations + 1):
        
        for individual in current_gen:
            if rnd.uniform(0, 1) < individual_completeness:
                # draw two parents from the population
                parents = rnd.choice(population_size, size=2, replace=False) + 1
                # make unique node ID
                parent_ID = parents + (population_size * t)

                for i in [0,1]:
                    G.add_node(parent_ID[i], generation=t,
                               x=parents[i], parents=[])
                    G.add_edge(individual, parent_ID[i])
                    G.nodes[individual]['parents'].append(parent_ID[i])
                    next_gen.add(parent_ID[i])
                

        # check that we have unique indexes
        assert(current_gen.intersection(next_gen) == set())
        
        # move on to t+1
        current_gen = next_gen.copy()
        next_gen = set()

    return G
        

def get_plotting_coord(G):
    return {k:(v['x'], v['generation']) for k,v in G.nodes(data=True)}
    

if __name__ == '__main__':
    g = simulate_wf_genealogy(5, 20, generations=3)

    fig, ax = plt.subplots(dpi=300)
    nx.draw(g, pos=get_plotting_coord(g),
            with_labels=True, node_shape='s', ax=ax)
    fig.savefig('fig/simulate.png')
