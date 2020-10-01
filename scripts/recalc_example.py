import numpy as np
import msprime as msp
from collections import Counter
from genealogy_aligner import Pedigree, DiploidGraph, Climber, Traversal
from genealogy_aligner.utils import invert_dictionary
from copy import deepcopy

seed = 2
P = Pedigree.simulate_from_founders_with_sex(10, 5, avg_immigrants=4, seed=seed)
G = DiploidGraph(P)
G_copy = deepcopy(G)
H = G.sample_haploid_path(seed)

C, msprime_idx = H.to_tree_sequence(simplify=True)
msp_labels = invert_dictionary(msprime_idx, one_to_one=True)
sim = msp.simulate(from_ts=C, Ne=1000, random_seed=seed, model="dtwf")
T = Traversal.from_tree_sequence(sim, msp_labels)
probands = T.probands()
assert (probands == G.probands())

tD = T.distance_matrix(kinship_like=True) / 2

# tree / genealogy pair
# agents = [(p,p) for p in probands]
parents = []
count = 0
choices = {}

times = G.get_node_attributes('time')
max_time = max(times.values())
depth_map = {t: [] for t in range(max_time+1)}
for p in probands:
    depth_map[0].append((p,p))
time = 0

while time < max_time:
    #TODO can we simplify the previous distance matrix without having to recalculate?
    gD = G.distance_matrix(kinship_like=True)
    scores = []

    agents = depth_map[time]
    for t_node, g_node in agents:
        t_parent =  T.parents(t_node)[0]

        if g_node not in G.graph.nodes: # we took a wrong turn somewhere
            continue
        g_parents = G.parents(g_node)
        if not g_parents:
            continue

        # get the current nodes being aligned
        # TODO: can we use previously aligned parents here too?
        t_nodes = [t for t,_ in agents]
        g_nodes = [g for _,g in agents]
        
        up_stat = tD[t_parent, t_nodes].todense()
        # calc with dot-products
        left  = gD[g_parents[0], g_nodes].todense() @ up_stat.T
        right = gD[g_parents[1], g_nodes].todense() @ up_stat.T if len(g_parents) == 2 else 0

        if left > right:
            scores.append((float(left), t_node, g_node, g_parents[0]))
        else:
            scores.append((float(right), t_node, g_node, g_parents[1]))

    count += 1
    if scores:
        ranking = sorted(scores, reverse=True)
        best_score = ranking[0][0]
        good_choices = [ch for ch in ranking if ch[0] == best_score]
    
        for score, t_node, g_node, g_parent in good_choices:
            depth_map[time].remove((t_node, g_node))
            G.graph.remove_node(g_node)
            print(count, score, t_node, g_node, g_parent)
            parents.append((score, t_node, g_node, g_parent))
    

    if not agents or not scores:
        
        parents = [p for p in parents if p[0] > 1e-10]

        merged_parents = set()
        # expand the parents with previous climbers - next generation
        for t_node, g_parent in depth_map[time + 1]:
            parents.append((-1, t_node, -1, g_parent))
            merged_parents.add(g_parent)
            
        for p in parents:
            print(p)
        parent_count = Counter(g for _,_,_,g in parents)

        for score, t_node, g_node, g_parent in parents:
            t = times[g_parent]
            # we can check if we are merging incorrectly here
            # if t_parents for some agents are not the same, but we assign same g_parent
            if parent_count[g_parent] > 1:
                # merge
                t_parent = T.parents(t_node)[0]
                if not g_parent in merged_parents:
                    merged_parents.add(g_parent)
                    depth_map[t].append((t_parent, g_parent))
                    # agents.append((t_parent, g_parent))
                    choices[t_parent] = g_parent
            else:
                if not g_parent in merged_parents:
                    # did not coalesce - keep climbing
                    depth_map[t].append((t_node, g_parent))
        
        time += 1

        print(f"t = {time}")
        for a in depth_map[time]:
            print(a)
        parents = []
        

incorrect = [(k,v) for (k,v) in choices.items() if k != v]
