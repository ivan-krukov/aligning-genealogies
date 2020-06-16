import networkx as nx
from genealogy_aligner import Pedigree, Traversal, Climber
import numpy as np
from collections import Counter
import numpy.random as rnd
import matplotlib.pyplot as plt


seed = rnd.randint(1000)
rnd.seed(seed)
print(f"Seed {seed}")
generations = 10
P = Pedigree.simulate_from_founders(200, generations, avg_immigrants=20)
depth = P.infer_depth(forward=False)

T = P.sample_haploid_path()
C = T.to_coalescent()
probands = C.probands()

dist = C.distance_matrix().todense()
dist[dist == 0] = np.inf
D = np.power(2.0, -dist)

K, idx = P.kinship_lange()

prob_idx = [idx[p] for p in probands]

climber = Climber(P, source=probands)
correct, incorrect, symmetries, total = Counter(), Counter(), Counter(), Counter()

for agent, pedigree_parents in climber:
    genealogy_parent = C.parent_of(agent)
    if not pedigree_parents:
        continue
    if not genealogy_parent:
        continue

    left_stat  = K[idx[pedigree_parents[0]], prob_idx]
    right_stat = K[idx[pedigree_parents[1]], prob_idx]

    up_stat = D[genealogy_parent, probands]

    left = up_stat @ left_stat
    right = up_stat @ right_stat

    d = depth[agent] + 1
    
    if left > right:
        choice = pedigree_parents[0]
    elif left < right:
        choice = pedigree_parents[1]
    else:
        rch = rnd.choice(2)
        # print(agent, ' draw - random choice - ', pedigree_parents[rch])
        choice = pedigree_parents[rch]
        symmetries[d] += 1

    climber.queue(choice)

    # add node-to-edge alignment
    if choice not in C.graph:
        C.graph.add_node(choice, inferred=True)
        C.graph.add_edge(choice, agent, inferred=True)
        C.graph.add_edge(genealogy_parent, choice, inferred=True)

    # check if correct
    
    try:
        if choice == T.parent_of(agent):
            correct[d] += 1
        else:
            incorrect[d] += 1
    except nx.NetworkXError:
        incorrect[d] += 1
    finally:
        total[d] += 1

print(correct)
print(incorrect)
print(symmetries)
print(total)

def make_hist(counter, max_key):
    bins = list(range(1, max_key+1))
    values = [0 for _ in bins]
    for i, b in enumerate(bins):
        values[i] = counter[b]
    return np.array(bins), np.array(values)

x, y_correct = make_hist(correct, generations)
x, y_incorrect = make_hist(incorrect, generations)
x, y_symmetries = make_hist(symmetries, generations)
x, y_total = make_hist(total, generations)

fig, ax = plt.subplots()
ax.set(ylim=(-0.05, 1.05), xlabel="Generation into past", ylabel="Percent", title=f"Seed {seed}, {generations} generations,  {P.n_individuals} in pedigree, {sum(y_total)} in genealogy")
ax.plot(x, y_correct / y_total, label = "Correct")
ax.plot(x, y_incorrect / y_total, label = "Incorrect")
ax.plot(x, y_symmetries / y_total, label = "Symmetries")

ax.legend()
fig.savefig('fig/agent_example.png', dpi=300)
