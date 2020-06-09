import networkx as nx
from genealogy_aligner import Pedigree, Traversal
import numpy as np
import numpy.random as rnd


rnd.seed(100)
P = Pedigree.simulate_from_founders(10, 5)

T = P.sample_haploid_path()
C = T.to_coalescent()
probands = C.probands()

dist = C.distance_matrix().todense()
dist[dist == 0] = np.inf
D = np.power(2.0, -dist)

K, idx = P.kinship_lange()

prob_idx = [idx[p] for p in probands]

R = Traversal()
R.graph.add_nodes_from(probands)

current_gen = set(probands)

while current_gen:
    next_gen = set()
    for agent in current_gen:
        print(agent)
        pedigree_parents = list(P.predecessors(agent))
        if not pedigree_parents:
            continue
        left_stat  = K[idx[pedigree_parents[0]], prob_idx]
        right_stat = K[idx[pedigree_parents[1]], prob_idx]

        genealogy_parent = C.parent_of(agent)
        if not genealogy_parent:
            continue
        
        up_stat = D[genealogy_parent, probands]

        left = up_stat @ left_stat
        right = up_stat @ right_stat

        random_choice = False
        if left > right:
            choice = pedigree_parents[0]
        elif left < right:
            choice = pedigree_parents[1]
        else:
            rch = rnd.choice(2)
            print(agent, ' draw - random choice - ', pedigree_parents[rch])
            random_choice = True
            choice = pedigree_parents[rch]
        if not random_choice:
            try:
                print(agent, T.predecessors(agent)[0] == choice)
            except:
                print(agent, False)

        if choice not in C.graph:
            C.graph.add_node(choice, inferred=True)
            C.graph.add_edge(choice, agent, inferred=True)
            C.graph.add_edge(genealogy_parent, choice, inferred=True)

        
        R.graph.add_node(choice)
        R.graph.add_edge(choice, agent)
        next_gen.add(choice)

    current_gen = next_gen


