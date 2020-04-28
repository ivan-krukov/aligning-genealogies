import numpy as np
from simulate import plot_graphviz, simulate_founder_genealogy
import networkx as nx
import matplotlib.pyplot as plt



def parents(G, i):
    return list(G.predecessors(i))



def kinship_matrix(G):
    n = len(G.nodes)

    K = np.zeros((n,n), dtype=float)

    for i in range(n):
        if parents(G, i):
            p, q = parents(G, i)
            K[i, i] = 0.5 + (K[p,q]/2)
        else:
           K[i,i] = 0.5
        for j in range(i+1, n):
            if parents(G, j):
                p, q = parents(G, j)
                v = (K[i,p]/2) + (K[i,q]/2)
                K[i,j] = v
                K[j,i] = v
            else:
                K[i, j] = 0
                K[j, i] = 0
    return K


def test_kinship_matrix():
    # Figure 5.1, Lange, 2ed (2003), p82
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 4)
    G.add_edge(3, 5)
    K = kinship_matrix(G)
    expected = np.array([[0.5  , 0.   , 0.25 , 0.25 , 0.25 , 0.25 ],
                         [0.   , 0.5  , 0.25 , 0.25 , 0.25 , 0.25 ],
                         [0.25 , 0.25 , 0.5  , 0.25 , 0.375, 0.375],
                         [0.25 , 0.25 , 0.25 , 0.5  , 0.375, 0.375],
                         [0.25 , 0.25 , 0.375, 0.375, 0.625, 0.375],
                         [0.25 , 0.25 , 0.375, 0.375, 0.375, 0.625]])
    assert np.allclose(K, expected)


def test_kinship_speed():
    families = 40
    generations = 10
    G = simulate_founder_genealogy(families, generations, 2.5)
    K = kinship_matrix(G)
    # for a genealogy with 4077 total individuals, the calculation takes 25s
