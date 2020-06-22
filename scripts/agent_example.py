import networkx as nx
from genealogy_aligner import *
import numpy as np
from collections import Counter
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy


seed = rnd.randint(1000)
rnd.seed(seed)
print(f"Seed {seed}")
founders = 50
generations = 10
P = Pedigree.simulate_from_founders_with_sex(founders, generations, avg_immigrants=20)
D = DiploidGraph(P)
depth = P.infer_depth(forward=False)

T = D.sample_haploid_path()
coal_nodes = nx.subgraph_view(T.graph, lambda n: T.graph.out_degree(n) > 1).nodes

C = T.to_coalescent()
probands = C.probands()

dist = C.distance_matrix().todense()
dist[dist == 0] = np.inf
coefficient = 2.0
Q = np.power(coefficient, -dist)

K, idx = D.kinship_lange(coefficient=coefficient)

prob_idx = [idx[p] for p in probands]


correct, incorrect, symmetries, total = Counter(), Counter(), Counter(), Counter()
agents = []

R = deepcopy(C)

# Climb the Pedigree starting at the probands
climber = Climber(D, source=probands)
for agent, pedigree_parents in climber:
    # get parent in the genealogy
    genealogy_parent = R.parent_of(agent)
    # do nothing if done
    if not pedigree_parents:
        continue
    if not genealogy_parent:
        continue

    # Get kinship statistics
    left_stat  = K[idx[pedigree_parents[0]], prob_idx]
    right_stat = K[idx[pedigree_parents[1]], prob_idx]
    # Get tree statistic
    up_stat = Q[genealogy_parent, probands]

    # Combine
    left = up_stat @ left_stat
    right = up_stat @ right_stat

    # bookkeeping
    d = depth[agent] + 1

    # Chose the larger combined statistic
    if left > right:
        choice = pedigree_parents[0]
    elif left < right:
        choice = pedigree_parents[1]
    else:
        rch = rnd.choice(2)
        choice = pedigree_parents[rch]
        symmetries[d] += 1

    # update the climber
    climber.queue(choice)

    # add node-to-edge alignment
    if choice not in R.graph:
        R.graph.add_node(choice, inferred=True)
        R.graph.add_edge(choice, agent, inferred=True)
        R.graph.add_edge(genealogy_parent, choice, inferred=True)

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

    # bookkeeping
    agents.append(agent)


assert len(agents) == len(set(agents))

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

plt.rc('grid', linestyle="--", color='0.8')
fig, ax = plt.subplots()
ax.set(ylim=(-0.05, 1.05), xlabel="Generation into past", ylabel="Percent", title=f"Seed {seed}, {generations} generations,  {P.n_individuals} in pedigree, {sum(y_total)} in genealogy")
ax.plot(x, y_correct / y_total, label = "Correct", marker='o')
ax.plot(x, y_incorrect / y_total, label = "Incorrect", marker='o')
ax.plot(x, y_symmetries / y_total, label = "Symmetries", marker='o')
ax.set_xticks(np.arange(1, generations + 1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend()
ax.grid(True)
fig.savefig('fig/agent_example.png', dpi=300)

# edge_inferred = R.get_edge_attr('inferred')
# node_inferred = R.get_node_attributes('inferred')
# node_colors = {node: ('green' if yes else 'red') for node, yes in node_inferred.items()}
# edge_colors = {node: ('green' if yes else 'red') for node, yes in edge_inferred.items()}
# R.draw(node_color=node_colors, edge_color=edge_colors)
