import networkx as nx
import numpy as np
import pandas as pd


def integer_dict(values, start=1):
    """Dict of `values` with keys in the integer range.
    Keys from `start` to `len(values)+start`, inclusive """

    n = len(values)
    return dict(zip(range(start, n + start), values))


def soft_ordering(total_edge_index, sim_scores):

    df = pd.DataFrame(np.concatenate((total_edge_index, sim_scores.reshape(1, -1))).T,
                      columns=['source', 'target', 'score'])

    df = df.sort_values(by=['source', 'score'], ascending=False)

    target_order = {}

    for s in df['source'].unique():
        sorted_targets = df.loc[df['source'] == s, 'target']
        target_order[s] = list(sorted_targets)

    return target_order


def greedy_matching(total_edge_index, sim_scores):

    pred_pairs = []

    while total_edge_index.shape[1] > 0:
        pair_idx = np.argmax(sim_scores)

        a, b = total_edge_index[:, pair_idx].flatten()
        pred_pairs.append((a, b))

        step_filt = (total_edge_index[0, :] != a) & (total_edge_index[1, :] != b)

        total_edge_index = total_edge_index[:, step_filt]
        sim_scores = sim_scores[step_filt]

    return pred_pairs


def draw_graphviz(G, labels=True, ax=None):
    """Uses `graphviz` to plot the genealogy"""
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos=pos, with_labels=labels, node_shape='s', ax=ax)

