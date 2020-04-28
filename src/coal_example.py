import networkx as nx
from pathlib import Path
import time
from src.simulate import plot_graphviz, simulate_wf_genealogy
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Generate a small genealogy, and produce a single realizeation tree from it
n = 5
G = simulate_wf_genealogy(n, n * 4, 7)

fig, ax = plt.subplots(dpi=300, figsize=(10,5))
plot_graphviz(G, ax)
fig.savefig('fig/shallow.png')

pairs = combinations(range(1, n+1), 2)


ancestral_trios = list(nx.all_pairs_lowest_common_ancestor(G, pairs))
founder = Counter(x[1] for x in ancestral_trios).most_common()[0][0]


T = nx.dfs_tree(G, founder)

fig, ax = plt.subplots(dpi=300, figsize=(10,5))
plot_graphviz(T, ax)
fig.savefig('fig/tree.png')
