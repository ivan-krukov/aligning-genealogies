import numpy as np
import msprime as msp
from collections import Counter
from genealogy_aligner import Pedigree, DiploidGraph, Climber, Traversal
from genealogy_aligner.utils import invert_dictionary
from copy import deepcopy

# this won't work in our case, since pedigree IDs are not contiguous
def permute_idx(M, idx):
    P = M.copy()
    idx_from = np.array(list(idx.keys()))
    idx_to   = np.array(list(idx.values()))
    P[idx_to] = M[idx_from]
    return P

seed = 1
P = Pedigree.simulate_from_founders_with_sex(4, 3, avg_immigrants=4, seed=seed)
G = DiploidGraph(P)
G_copy = deepcopy(G)
H = G.sample_haploid_path(seed)

C, msprime_idx = H.to_tree_sequence(simplify=True)
msp_labels = invert_dictionary(msprime_idx, one_to_one=True)
sim = msp.simulate(from_ts=C, Ne=1000, random_seed=seed, model="dtwf")
T = Traversal.from_tree_sequence(sim, msp_labels)
probands = T.probands()
assert (probands == G.probands())

# TODO this is not giving the correct kinship for siblings
tD = T.distance_matrix(kinship_like=True) / 2

# tree / genealogy pair
agents = [(p,p) for p in probands]
parents = []
keep_climbing = True
count = 0
choices = {}
# TODO: ignore really bad decisions - score < 1e-9
while keep_climbing:
    gD = G.distance_matrix(kinship_like=True)
    scores = []
    
    for t_node, g_node in agents:
        t_parent =  T.parents(t_node)[0]

        g_parents = G.parents(g_node)
        if not g_parents:
            continue

        t_nodes = [t for t,g in agents]
        g_nodes = [g for t,g in agents]
        up_stat = tD[t_parent, t_nodes].todense()

        # calc with dot-products
        left  = gD[g_parents[0], g_nodes].todense() @ up_stat.T
        right = gD[g_parents[1], g_nodes].todense() @ up_stat.T if len(g_parents) == 2 else 0

        if left > right:
            scores.append((float(left), t_node, g_node, g_parents[0]))
        else:
            scores.append((float(right), t_node, g_node, g_parents[1]))

    count += 1
    if not scores:
        break
    ranking = sorted(scores, reverse=True)
    best_score, best_t_node, best_g_node, best_g_parent = ranking[0]
    agents.remove((best_t_node, best_g_node))
    G.graph.remove_node(best_g_node)
    print(count, ranking[0:max(len(ranking)-1,1)])

    parents.append((best_t_node, best_g_parent))
    
    if best_score > 1e-10:
        choices[best_g_node] = best_t_node
    

    if not agents:
        # TODO merge parents
        parent_count = Counter(g for t,g in parents)
        merged_parents = set()
        agents = []
        for t_node, g_parent in parents:
            if parent_count[g_parent] > 1:
                # merge
                t_parent = T.parents(t_node)[0]
                if not g_parent in merged_parents:
                    merged_parents.add(g_parent)
                    agents.append((t_parent, g_parent))
            else:
                agents.append((t_node, g_parent))
        print(agents)
        parents = []

print(choices)
