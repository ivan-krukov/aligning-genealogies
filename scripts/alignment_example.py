import numpy as np
from tqdm import tqdm
import networkx as nx
from genealogy_aligner import Pedigree, Traversal
from genealogy_aligner.kinship import kinship_matrix
import matplotlib.pyplot as plt
from numpy import random as rnd
import copy
import pickle

def distances(T):
    return dict(nx.all_pairs_shortest_path_length(nx.to_undirected(T.graph)))



# P = Pedigree.read_balsac("data/simple.tsv")
# probands = [8, 9, 10]

# t = nx.DiGraph()
# t.add_nodes_from(
#     [
#         #(7, {"time": 0}),
#         (8, {"time": 0}),
#         (9, {"time": 0}),
#         (10, {"time": 0}),
#         (4, {"time": 1}),
#         (5, {"time": 1}),
#         (1, {"time": 2}),
#     ]
# )
# t.add_edges_from([[5, 10], [5, 9], [1, 4], [4, 8], [1, 5]])
# T = Traversal(t)

P = Pedigree.simulate_from_founders(6, 3, 2, 5)
probands = P.probands()
T = P.sample_path(probands)
coal = T.to_coalescent_tree()


C = copy.deepcopy(coal)
nx.set_node_attributes(C.graph, False, 'inferred')
nx.set_edge_attributes(C.graph, False, 'inferred')

index = dict(zip(P.nodes, [P.nodes.index(n) for n in P.nodes]))
prob_idx = [P.nodes.index(n) for n in probands]  # make sure it's the same in the tree



dist = distances(C)
tbl = P.to_table()
K = kinship_matrix(tbl.individual, tbl.mother, tbl.father, tbl.time)


def get_similarity(dist, node, probands):
    d = np.array([dist[node].get(p, np.inf) for p in probands])
    return np.power(2.0, -d)

R = Traversal()
R.graph.add_nodes_from(probands)

current_gen = set(probands)
next_gen = set()

while current_gen:
    for agent in current_gen:
        print(agent)
        pedigree_parents = list(P.predecessors(agent))
        if not pedigree_parents:
            continue
        idx = [P.nodes.index(n) for n in current_gen]
        left_stat  = K[index[pedigree_parents[0]], prob_idx]
        right_stat = K[index[pedigree_parents[1]], prob_idx]

        genealogy_parent = list(C.predecessors(agent))[0]
        up_stat = get_similarity(dist, genealogy_parent, probands)

        left = up_stat @ left_stat
        right = up_stat @ right_stat
        
        # print(agent, genealogy_parent, up_stat, pedigree_parents[0], left, left_stat, pedigree_parents[1], right, right_stat)
        

        random_choice = False
        if left > right:
            choice = pedigree_parents[0]
        elif left < right:
            choice = pedigree_parents[1]
        else:
            rch = rnd.choice(2)
            print(agent, ' draw - random choice - ', pedigree_parents[rch])
            random_choice = True
            choice = pedigree_parents[rch]
        if not random_choice:
            try:
                print(agent, T.predecessors(agent)[0] == choice)
            except:
                print(agent, False)

        if choice not in C.graph:
            C.graph.add_node(choice, inferred=True)
            C.graph.add_edge(choice, agent, inferred=True)
            C.graph.add_edge(genealogy_parent, choice, inferred=True)
        
        
        R.graph.add_node(choice)
        R.graph.add_edge(choice, agent)
        next_gen.add(choice)
            
        

    current_gen = next_gen
    next_gen = set()

real = nx.subgraph_view(C.graph, lambda x: C.graph.nodes[x]['inferred'] == False).nodes
inferred = nx.subgraph_view(C.graph, lambda x: C.graph.nodes[x]['inferred'] == True).nodes
inferred_edges = nx.subgraph_view(C.graph, filter_edge=lambda a,b: C.graph.edges[a, b]['inferred'] == True).edges
real_edges = nx.subgraph_view(C.graph, filter_edge=lambda a,b: C.graph.edges[a, b]['inferred'] == False).edges


fig, ax = plt.subplots(ncols=4, figsize=(20,8), dpi=100)
# pos = nx.drawing.nx_agraph.graphviz_layout(C.graph, prog='dot')
# nx.draw_networkx_nodes(C.graph, pos, nodelist=real, ax=ax[0], node_shape='s')
# nx.draw_networkx_labels(C.graph, pos, nodelist=real, ax=ax[0], font_color='white', font_size=8)
# nx.draw_networkx_nodes(C.graph, pos, nodelist=inferred, node_color='r', ax=ax[0], node_shape='s')
# nx.draw_networkx_labels(C.graph, pos, nodelist=inferred, ax=ax[0], font_color='white', font_size=8)
# nx.draw_networkx_edges(C.graph, pos, ax=ax[0])

R.draw(ax=ax[0],frame_on=True)
T.draw(ax=ax[1])
coal.draw(ax=ax[2])
P.draw(ax=ax[3])
fig.savefig('fig/alignement_example.png')
