import networkx as nx
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from itertools import count


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
    
    G = nx.DiGraph() # should this be directed?

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
                    G.add_edge(parent_ID[i], individual)
                    G.nodes[individual]['parents'].append(parent_ID[i])
                    next_gen.add(parent_ID[i])
                

        # check that we have unique indexes
        assert(current_gen.intersection(next_gen) == set())
        
        # move on to t+1
        current_gen = next_gen.copy()
        next_gen = set()

    return G


def simulate_founder_genealogy(families, generations, mean_offspring=2):
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
    G = nx.DiGraph()

    current_gen = []
    next_gen = []
    # insert founder families
    for f in range(families):
        mat_id, pat_id = 2*f, 2*f+1
        G.add_node(mat_id, time=0)
        G.add_node(pat_id, time=0)
        current_gen.extend([mat_id, pat_id])

    id_counter = count(families*2)

    for t in range(generations):
        # if one individual is left, it produces no offspring
        while len(current_gen) >= 2: 
            mat_id, pat_id = rnd.choice(current_gen, 2, replace=False)
            current_gen.remove(mat_id)
            current_gen.remove(pat_id)
            children = rnd.poisson(mean_offspring)

            for ch in range(children):
                child_id = next(id_counter)
                next_gen.append(child_id)
                G.add_node(child_id, time=t)
                G.add_edge(mat_id, child_id)
                G.add_edge(pat_id, child_id)
        current_gen = next_gen
        next_gen = []

    return G


def get_plotting_coord(G):
    return {k:(v['x'], v['generation']) for k,v in G.nodes(data=True)}


def plot_graphviz(G, ax = None, labels=True):
    # this requires `graphviz` and `pygraphviz`
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos, with_labels=labels, node_shape='s', ax=ax)
    

if __name__ == '__main__':

    ## simulation
    g = simulate_wf_genealogy(10, 1000, generations=6, individual_completeness=0.9)

    ## plot example
    fig, ax = plt.subplots(dpi=300, figsize=(10,5))
    plot_graphviz(g, ax)
    fig.savefig('fig/simulate.png')
    
    ## export example
    nx.write_adjlist(g, 'cached/simulation_1.txt', delimiter='\t')
    h = nx.read_adjlist('cached/simulation_1.txt', nodetype=int)
    # roundtrip
    assert(sorted(h.nodes) == sorted(g.nodes))
    # assert(sorted(h.edges) == sorted(g.edges)) # this needs a deep comparison
