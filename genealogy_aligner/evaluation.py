from utils import greedy_matching, soft_ordering
import numpy as np


def accuracy(pos_anchors, pred_anchors):
    """
    Defined as in Equation (13) of Trung et al.
    https://www.sciencedirect.com/science/article/pii/S0957417419305937

    :param pos_edge_index: list of tuples of matching nodes in graphs 1 and 2
    :param pred_edge_index: list of tuples of predicted matching nodes in graphs 1 and 2

    """
    return float(len(set(pos_anchors).intersection(set(pred_anchors)))) / len(pos_anchors)

def MAP(pairs, sim_score):
    s_order = soft_ordering(pairs, sim_score)
    return np.mean([1. / (li.index(n) + 1) for n, li in s_order.items() if n in li])

