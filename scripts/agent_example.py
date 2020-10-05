import networkx as nx
from genealogy_aligner import *
import numpy as np
from collections import Counter
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy
from genealogy_aligner.utils import invert_dictionary
import msprime as msp
from scipy.spatial.distance import cdist
import numpy.linalg as la


def ancestor_chain(T, node):
    chain = []
    p = T.first_parent_of(node)
    while p is not None:
        chain.append(p)
        node = p
        p = T.first_parent_of(node)
    return chain


class Agent:
    def __init__(self, ancestor_chain, ref_count=1):
        self.ancestor_chain = ancestor_chain
        self.ref_count = ref_count

    def __repr__(self):
        return f"Agent({self.ref_count}, {self.ancestor_chain})"


def tanimoto(a, b):
    dot = a @ b
    an = la.norm(a)
    bn = la.norm(b)
    return dot / (an ** 2 + bn ** 2 - dot)


def cosine(a, b):
    dot = a @ b
    an = la.norm(a)
    bn = la.norm(b)
    return dot / (an * bn)


seed = rnd.randint(1000)
seed = 1
rnd.seed(seed)
print(f"Seed {seed}")
founders = 10
generations = 7
immigrants = 4
progress = True

P = Pedigree.simulate_from_founders_with_sex(
    founders, generations, avg_immigrants=immigrants, seed=seed
)
D = DiploidGraph(P)
depth = D.get_node_attributes("time")

T = D.sample_haploid_path(seed)
# largest = max(nx.weakly_connected_components(T.graph), key=len)
# T.graph = nx.subgraph(T.graph, largest)
C, msprime_idx = T.to_tree_sequence(simplify=True)
msp_labels = invert_dictionary(msprime_idx, one_to_one=True)
M = msp.simulate(from_ts=C, Ne=1000, random_seed=seed, model="dtwf")
O = Traversal.from_tree_sequence(M, msp_labels)
probands = O.probands()

Q = O.distance_matrix(kinship_like=True, progress=progress) / 2

K, idx = D.kinship_lange(progress=progress)



correct = Counter()
incorrect = Counter()
symmetries = Counter()
total = Counter()

agents = {}
for p in probands:
    ancestors = ancestor_chain(O, p)
    agent = Agent(ancestors)
    agents[p] = agent
agent_nodes = probands
choices = {}
symmetries = Counter()
climber = Climber(D, source=probands)
for ped_node, ped_parents in climber:
    agent_idx = [idx[p] for p in agent_nodes]

    if not ped_parents:
        continue

    agent = agents[ped_node]
    if agent.ref_count > 1:
        agent.ancestor_chain.pop(0)

    gen_parent = agent.ancestor_chain[0]

    d = depth[ped_node] + 1

    left_stat = K[idx[ped_parents[0]], agent_idx]
    right_stat = K[idx[ped_parents[1]], agent_idx]
    up_stat = Q[gen_parent, agent_nodes].todense()

    # calc with dot-products
    left = up_stat @ left_stat
    right = up_stat @ right_stat

    # calc with alt similarity
    # left = tanimoto(up_stat, left_stat)
    # right = tanimoto(up_stat, right_stat)

    if left > right:
        choice = ped_parents[0]
    elif left < right:
        choice = ped_parents[1]
    else:
        rch = rnd.choice(2)
        choice = ped_parents[rch]
        symmetries[d] += 1
    choices[ped_node] = choice
    agent_nodes.remove(ped_node)
    agent_nodes.append(choice)

    climber.queue(choice)

    if choice in agents:
        agents[choice].ref_count += 1
    else:
        a = Agent(agent.ancestor_chain)
        agents[choice] = a

correct_ind = Counter()
correct_ploid = Counter()
total = Counter()
for node, inferred_parent in choices.items():
    d = depth[node] + 1
    try:
        true_parent = T.first_parent_of(node)
        inferred_ind = (inferred_parent + 1) // 2
        true_ind = (true_parent + 1) // 2
        if inferred_ind == true_ind:
            correct_ind[d] += 1
        if inferred_parent == true_parent:
            correct_ploid[d] += 1
    except:
        pass
    finally:
        total[d] += 1


def make_hist(counter, max_key):
    bins = list(range(1, max_key + 1))
    values = [0 for _ in bins]
    for i, b in enumerate(bins):
        values[i] = counter[b]
    return np.array(bins), np.array(values)


x, y_correct_ind = make_hist(correct_ind, generations)
x, y_correct_ploid = make_hist(correct_ploid, generations)
x, y_symmetries = make_hist(symmetries, generations)
x, y_total = make_hist(total, generations)

plt.rc("grid", linestyle="--", color="0.8")
fig, ax = plt.subplots()
ax.set(
    ylim=(-0.05, 1.05),
    xlabel="Generation into past",
    ylabel="Percent",
    title=f"Seed {seed}, {generations} generations,  {P.n_individuals} in pedigree, {sum(y_total)} in genealogy",
)
ax.plot(x, y_correct_ind / y_total, label="Correct individual", marker="o")
ax.plot(x, y_correct_ploid / y_total, label="Correct ploid", marker="o")
ax.plot(x, y_symmetries / y_total, label="Symmetries", marker="o")
ax.set_xticks(np.arange(1, generations + 1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.legend()
ax.grid(True)
fig.savefig("fig/agent_alignment.png", dpi=300)
