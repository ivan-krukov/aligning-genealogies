import networkx as nx
import numpy as np
from Genealogy import Genealogy
from utils import draw_graphviz
import matplotlib.pyplot as plt
import numpy as np

G = Genealogy.from_founders(families=3,
                            generations=3,
                            mean_offspring=2,
                            mean_out_of_family=1)
T = G.sample_path()


# relabel T
S = nx.convert_node_labels_to_integers(T, ordering='sorted')
TS_mapping = dict(zip(T.nodes, S.nodes))
ST_mapping = dict(zip(S.nodes, T.nodes))

s_nodes = list(S.nodes())
s_nodes_round = [TS_mapping[t] for t in [ST_mapping[s] for s in s_nodes]]
assert s_nodes == s_nodes_round

# save
nx.write_adjlist(G, 'cached/G_working.txt')
nx.write_adjlist(T, 'cached/T_working.txt')
nx.write_adjlist(S, 'cached/S_working.txt')

s = S.similarity()
k = G.kinship()
probands = G.probands()
leaves = S.probands()

kinship_to_probands = k[:probands[0], probands[0]:]

symmetries = 0
for i in range(kinship_to_probands.shape[0]):
    row = kinship_to_probands[i, :]
    sym  = np.where(np.all(row == kinship_to_probands, axis=1))[0]
    if len(sym) > 1:
        symmetries += 1
print(symmetries / kinship_to_probands.shape[0])
