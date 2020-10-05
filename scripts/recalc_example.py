import numpy as np
import numpy.random as rnd
import msprime as msp
from collections import Counter, defaultdict
from genealogy_aligner import (
    Pedigree,
    DiploidGraph,
    Climber,
    Traversal,
    Agent,
    AgentSet,
)
from genealogy_aligner.utils import invert_dictionary
from copy import deepcopy
import networkx as nx
from operator import itemgetter


seed = 1
# P = Pedigree.simulate_from_founders_with_sex(20, 7, avg_immigrants=10, seed=seed)
P = Pedigree.simulate_from_founders_with_sex(4, 4, avg_immigrants=1, seed=seed)

print(P.n_individuals)
# P = Pedigree.from_balsac_table("data/test/example_3.tsv")
G = DiploidGraph(P)
nx.set_node_attributes(G.graph, G.infer_depth(forward=False), "depth")

G_copy = deepcopy(G)
# probands = [21, 25,28]
probands = G.probands()
H = G.sample_haploid_path(seed, probands)

C, msprime_idx = H.to_tree_sequence(simplify=True)
msp_labels = invert_dictionary(msprime_idx, one_to_one=True)
sim = msp.simulate(from_ts=C, Ne=1000, random_seed=seed, model="dtwf")
T = Traversal.from_tree_sequence(sim, msp_labels)

tD = T.distance_matrix(kinship_like=True)
# gD = G.distance_matrix(kinship_like=True)
K, idx = G.kinship_lange()

count = 0
choices = defaultdict(list)

# tree / genealogy

agents = AgentSet(G, T)
for p in probands:
    agents.add(Agent(p, p, G, T, 1))

time = 0


while time < G.generations:
    scores = []
    for a in agents.at(time):
        # t_parent =  T.parents(t_node)[0]

        if a.g_node not in G.graph.nodes:  # we took a wrong turn somewhere
            continue
        if not a.g_parents():
            continue
        if not a.t_parent():
            continue

        up_stat = tD[a.t_parent(), agents.t_nodes()].todense()
        # calc with dot-products
        agent_idx = [idx[p] for p in agents.g_nodes()]

        parent_scores = (
            float(K[idx[a.g_parents()[0]], agent_idx] @ up_stat.T),
            float(K[idx[a.g_parents()[1]], agent_idx] @ up_stat.T),
        )

        if parent_scores[0] > parent_scores[1]:
            scores.append((parent_scores[0], a, a.g_parents()[0]))
        elif parent_scores[0] < parent_scores[1]:
            scores.append((parent_scores[1], a, a.g_parents()[1]))
        else:
            # print("random choice")
            rch = rnd.choice(2)
            scores.append((parent_scores[rch], a, a.g_parents()[rch]))

    count += 1
    if scores:
        ranking = sorted(scores, reverse=True, key=itemgetter(0))
        best_score = ranking[0][0]

        good_choices = [ch for ch in ranking if ch[0] == best_score]
        for score, a, g_parent in good_choices:
            b = Agent(g_parent, a.t_node, G, T, score)
            agents.remove(a)
            if score > 0:
                agents.add(b)
            # print(count, "{:.2}".format(score), a, "\t", b)

    if not agents.at(time) or not scores:

        print("merging")
        merge_candidates = sorted(
            agents.at(time + 1), reverse=True, key=lambda c: c.score
        )
        visited = {}
        updated = set()  # make sure we only advance the parent pointer once
        for a in merge_candidates:
            print(a)
            if a.g_node not in visited.keys():
                visited[a.g_node] = a
            else:
                b = visited[a.g_node]
                assert a.g_node == b.g_node
                # merge
                if a.g_node not in updated:
                    # only update once - guaranteed best score since we sorted
                    print("> ", a, b)
                    visited[a.g_node].t_node = b.t_parent()  # advance by pointer
                    choices[a.g_node].append(b)
                    updated.add(a.g_node)

                agents.remove(a)

        print(agents.all())
        time += 1
        print(f"t = {time}")
        parents = []


correct_chromosome = defaultdict(int)
correct_individual = defaultdict(int)
off_by_one = defaultdict(int)
total = defaultdict(int)

for g_node, candidates in choices.items():
    print(g_node)
    print(candidates)
    # d = agent.depth()

    # cc = agent.t_node == g_node
    # ci = cc or (agent.t_node == (g_node - 1))
    # correct_chromosome[d] += cc
    # correct_individual[d] += ci
    # total[d] += 1

    # print("\t", agent)
