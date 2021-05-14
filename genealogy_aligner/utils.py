import networkx as nx
from collections.abc import Iterable
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


def dict_to_csr(d, items):
    """
    Takes a dictionary of dictionaries where the first key is the row name
    and the second key is the column name and converts into a scipy
    sparse matrix format.
    """

    item_idx = dict(zip(items, range(len(items))))

    row, col, data = [], [], []

    for k1 in d:
        row += list(np.repeat(item_idx[k1], len(d[k1])))
        col += [item_idx[k2] for k2 in d[k1]]
        data += list(d[k1].values())

    return csr_matrix((data, (row, col)), shape=(len(items), len(items)))


def invert_dictionary(d):
    """
    Takes a dictionary returns its inverse (values as keys and keys as values).
    To account for duplicate values, it returns the keys as a list.
    """

    inv_dict = dict()

    for k, v in d.items():
        inv_dict.setdefault(v, []).append(k)

    return inv_dict


def create_attr_dictionary(items, attr_dict, default, invert=False):
    """
    A helper function that takes a list of items, a partial dictionary containing
    information for some of the items, and a default value for the remaining. The
    function fills in the value for the remaining items.
    If `invert` is `True`, the dictionary is inverted so that values are keys and
    keys are values.
    :param items:
    :param attr_dict:
    :param default:
    :param invert:
    :return:
    """
    fin_attr_dict = dict(zip(items, [default] * len(items)))

    if attr_dict is not None:
        fin_attr_dict.update(attr_dict)

    fin_attr_dict = {k: v for k, v in fin_attr_dict.items() if k in items}

    if invert:
        return invert_dictionary(fin_attr_dict)
    else:
        return fin_attr_dict


def get_k_order_neighbors(graph, source_nodes,
                          radius=1, undirected=True):
    """
    This function takes a graph, a source node, and a radius,
    and returns a dictionary of all nodes in the graph that are
    within that radius as well as the (min) distance from the nodes
    in the subset.

    :param graph:
    :param source_nodes:
    :param radius:
    :param undirected:
    :return:
    """

    if undirected:
        graph = graph.to_undirected()

    if not isinstance(source_nodes, Iterable) or type(source_nodes) == str:
        source_nodes = [source_nodes]

    node_subset = {}

    for n in source_nodes:
        for nn, dist in nx.single_source_shortest_path_length(graph, n,
                                                              cutoff=radius).items():
            if nn in node_subset:
                node_subset[nn] = min(node_subset[nn], dist)
            else:
                node_subset[nn] = dist

    return node_subset


def integer_dict(values, start=1):
    """Dict of `values` with keys in the integer range.
    Keys from `start` to `len(values)+start`, inclusive """

    n = len(values)
    return dict(zip(range(start, n + start), values))


def soft_ordering(total_edge_index, sim_scores, return_confidence=False):

    df = pd.DataFrame(np.concatenate((total_edge_index, sim_scores.reshape(1, -1))).T,
                      columns=['source', 'target', 'score'])

    df = df.sort_values(by=['source', 'score'], ascending=False)

    target_order = {}

    for s in df['source'].unique():
        if return_confidence:
            sorted_targets = df.loc[df['source'] == s, ['target', 'score']]
            sorted_targets['score'] /= sorted_targets['score'].sum()
            target_order[s] = list(sorted_targets.to_records(index=False))
        else:
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

