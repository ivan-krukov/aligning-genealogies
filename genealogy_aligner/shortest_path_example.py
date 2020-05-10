import networkx as nx
from pathlib import Path
import time
from Genealogy import Genealogy
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter


G = Genealogy.from_founders(5, 5, 2.5)
U = G.graph.to_undirected()


start = time.perf_counter()
for a, b in combinations(G.probands(), 2):
    try:
        paths = nx.all_shortest_paths(U, a, b)
        print(list(paths))
    except nx.NetworkXNoPath:
        print(f'no path between {a}, {b}')
    
print(time.perf_counter() - start)





