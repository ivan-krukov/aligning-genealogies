import networkx as nx
from Genealogy import Genealogy
from utils import draw_graphviz
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# Generate a small genealogy, and produce a single realizeation tree from it
# 5 finding families, 5 generations
n = 5
G = Genealogy.from_founders(n, 5)

# plot
fig, ax = plt.subplots(dpi=300, figsize=(10,5))
G.draw(ax)
fig.savefig('fig/coal_example_genealogy.png')

# every pair of probands
pairs = combinations(G.probands(), 2)

# find a set of common ancestors
ancestral_trios = list(nx.all_pairs_lowest_common_ancestor(nx.DiGraph(G), pairs))

# find the most common common ancestor
important_ancestor = Counter(x[1] for x in ancestral_trios).most_common()[0][0]

# plot
T = nx.dfs_tree(G, important_ancestor)
fig, ax = plt.subplots(dpi=300, figsize=(10,5))
draw_graphviz(T, ax)
fig.savefig('fig/coal_example_tree.png')
