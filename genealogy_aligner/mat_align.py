import networkx as nx
import numpy as np
from Genealogy import Genealogy
import matplotlib.pyplot as plt

G = Genealogy.from_founders(families=10,
                            generations=7,
                            mean_offspring=2,
                            mean_out_of_family=2)
T = G.sample_path()
p = G.probands()

k = G.kinship()
s = T.similarity(G)

match = 0
for i in T:
    match += (i == np.argmax(s[i,:] @ k))

print(match / len(T))
